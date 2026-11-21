"""Evaluate a signal-triggered, frozen MetricGAN+ front end on the baseline ASR.

Choose the noise-floor threshold on training audio only, then use automatic ASR on
the same validation views as the LoRA experiment. Unchanged inputs reuse exact
baseline results. No test output participates in choosing the front end.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import periodogram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cleartalk_ml.text import counts
from scripts.train_three_language_asr import split_rows, wave
from scripts.validate_pitch_asr import (
    INCUMBENT,
    AuditPipeline,
    validate_sets,
    read_rows,
    summary,
    macro_noisy,
    paired_interval,
    condition_audio,
    noise_pool,
    LANGS,
)

REVISION = "a196ce26b3bdace6fa1d819017584bdbcce462a8"
MODEL = ROOT / "models/pitch-enhancement"


def floor_db(audio):
    frame = 400
    n = len(audio) // frame
    if not n or np.mean(audio**2) < 1e-10:
        return -100.0
    frames = audio[: n * frame].reshape(n, frame)
    _, spec = periodogram(
        frames, fs=16000, window="hann", scaling="density", detrend=False
    )
    band_power = spec[:, 100:190].sum(axis=1) * 40
    return float(
        10
        * np.log10(max(float(np.quantile(band_power, 0.1)), 1e-12) / np.mean(audio**2))
    )


def fit_gate():
    training = split_rows("train", 48, "stationary-noise-gate")
    reserved = {r["utt_id"] for r in noise_pool()}
    reserved.update(r["utt_id"] for rs in training.values() for r in rs)
    pool = split_rows("train", 80, "stationary-noise-background")
    background = [r for rs in pool.values() for r in rs if r["utt_id"] not in reserved]
    negatives = []
    for rs in training.values():
        for row in rs:
            uid = row["utt_id"]
            x = wave(row)
            negatives.extend(
                (
                    floor_db(x),
                    floor_db(condition_audio(x, uid, "babble-10", background)),
                )
            )
    return float(np.quantile(negatives, 0.99))


class Enhancement:
    def __init__(self, threshold):
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        self.threshold = threshold
        self.net = SpectralMaskEnhancement.from_hparams(
            source=str(MODEL), savedir=str(MODEL), run_opts={"device": "cpu"}
        )

    @torch.inference_mode()
    def apply(self, x):
        if floor_db(x) < self.threshold:
            return x
        y = (
            self.net.enhance_batch(torch.from_numpy(x)[None], lengths=torch.ones(1))[0]
            .cpu()
            .numpy()
        )
        # Preserve the original sample grid and retain a small dry component.
        out = x.copy()
        n = min(len(y), len(x))
        out[:n] = 0.8 * y[:n] + 0.2 * x[:n]
        assert np.isfinite(out).all()
        return out


def score(sets, baseline, path, threshold, threads):
    torch.set_num_threads(threads)
    enhancer = Enhancement(threshold)
    pipe = AuditPipeline(INCUMBENT, threads, path.parent / "detector")
    bg = noise_pool()
    rows = read_rows(path)
    for lang, rs in sets.items():
        for uid, x, ref in rs:
            for condition in ("clean", "white-10", "babble-10"):
                key = uid, condition
                if key in rows:
                    continue
                original = condition_audio(x, uid, condition, bg)
                enhanced = enhancer.apply(original)
                if enhanced is original:
                    entry = dict(baseline[key], enhanced=False)
                else:
                    pipe.detector_votes.clear()
                    pipe.used_candidate = False
                    result = pipe.transcribe(enhanced, language="auto", speakers=1)
                    turns = result["turns"]
                    hyp = " ".join(t["text"] for t in turns)
                    entry = dict(
                        uid=uid,
                        language=lang,
                        condition=condition,
                        enhanced=True,
                        exact_lid={t["language"] for t in turns} == {lang},
                        **counts(ref, hyp),
                    )
                    if path.stem == "test":
                        entry.update(
                            hypothesis=hyp, detector_votes=list(pipe.detector_votes)
                        )
                rows[key] = entry
                with path.open("a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(len(rows), lang, condition, entry["enhanced"], flush=True)
    assert rows.keys() == baseline.keys()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.fetch:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "speechbrain/metricgan-plus-voicebank",
            revision=REVISION,
            local_dir=MODEL,
            allow_patterns=["enhance_model.ckpt", "hyperparams.yaml", "README.md"],
            token=False,
        )
    work = ROOT / "models/pitch-asr"
    folder = work / "stationary-noise"
    folder.mkdir(exist_ok=True)
    decision = folder / "selection.json"
    if args.test:
        from scripts.final_pitch_controls import controls

        chosen = json.loads(decision.read_text())
        if not chosen["eligible"]:
            raise SystemExit("Validation retained the original audio")
        base = read_rows(work / "holdout/0.jsonl")
        rows = score(
            controls(),
            base,
            folder / "test.jsonl",
            chosen["threshold_db"],
            args.threads,
        )
        (folder / "test.json").write_text(json.dumps(summary(rows), indent=2) + "\n")
        return
    threshold = fit_gate()
    base = read_rows(work / "validation/0.jsonl")
    assert len(base) == 108
    rows = score(
        validate_sets(), base, folder / "validation.jsonl", threshold, args.threads
    )
    old, new = summary(base), summary(rows)
    ci = paired_interval(base, rows)
    clean_safe = all(
        new[l + "_clean"]["wer"] <= old[l + "_clean"]["wer"] + 0.005 for l in LANGS
    )
    macro = float(macro_noisy(new))
    eligible = bool(clean_safe and ci[1] < 0 and macro <= 0.9 * macro_noisy(old))
    result = dict(
        threshold_db=threshold,
        eligible=eligible,
        clean_safe=clean_safe,
        macro_noisy_wer=macro,
        baseline_macro_noisy_wer=float(macro_noisy(old)),
        delta_ci95=ci,
        train_sources=144,
        enhanced_views=sum(r["enhanced"] for r in rows.values()),
        validation=new,
    )
    decision.write_text(json.dumps(result, indent=2) + "\n")
    print(result["eligible"], flush=True)


if __name__ == "__main__":
    main()
