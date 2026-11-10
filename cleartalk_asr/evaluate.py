"""Transcribe evaluation tracks and score them.

    python -m cleartalk_asr.evaluate --set val --model baseline --engine app
    python -m cleartalk_asr.evaluate --set val --model models/whisper-small.en-cleartalk --engine app

``engine=app`` is the deployed path: faster-whisper int8 on CPU with the app's exact
decoding policy (beam 5, speech detection). ``engine=hf`` is greedy decoding with the
PyTorch model and is used for fast checks during training. Test sets are only scored
through ``ml_pipeline_english/04_eval.py`` (guarded, once).
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

from .mixtures import load_tracks
from .paths import BASE_WHISPER_CT2, DATA, MIXTURES, ROOT
from .text import add_counts, edit_counts, pair_counts

RESULTS = DATA / "results"


# ----------------------------------------------------------------------------- evaluation sets
def v4_tracks(split):
    """Export the earlier v4 cached separator outputs (48 validation / 96 test mixtures) as WAV tracks."""
    name = {"validation": "v4val", "test": "v4test"}[split]
    manifest = json.loads((ROOT / f"data/v4/asr_{'val' if split == 'validation' else 'test'}_manifest.json").read_text())
    out_dir = MIXTURES / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in manifest["pairs"]:
        cache = ROOT / f"archive/evaluation/v4/asr_{split}/audio/{record['id']}.npz"
        with np.load(cache) as audio:
            separated = audio["baseline"]
        for k in range(2):
            path = out_dir / f"{record['id']}-{k}.wav"
            if not path.exists():
                track = separated[:, k].astype(np.float32).copy()
                peak = float(np.max(np.abs(track)))
                if peak > 1e-6:
                    track *= 0.95 / peak
                sf.write(path, track, 8000, subtype="PCM_16")
            source = record["sources"][k]
            rows.append({"track_id": f"{record['id']}-{k}", "recipe_id": record["id"], "kind": "pair",
                         "path": path.relative_to(ROOT).as_posix(), "text_ls": source["reference_text"],
                         "speaker": source["speaker"], "noise": record["noise"], "snr_db": record["snr_db"],
                         "reverb": record["reverb"], "delay": record["delay_fraction"], "si_sdr_db": None})
    return pd.DataFrame(rows)


def eval_tracks(set_name):
    if set_name in ("v4val", "v4test"):
        return v4_tracks("validation" if set_name == "v4val" else "test")
    return load_tracks(set_name).sort_values("track_id").reset_index(drop=True)


# ----------------------------------------------------------------------------- transcription
def transcribe_app(model_dir, paths, threads=4):
    """The deployed path: faster-whisper int8 + backend/transcription.py policy."""
    sys.path.insert(0, str(ROOT))
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    from backend.transcription import transcribe_audio

    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=threads,
                         num_workers=1, local_files_only=True)
    texts = []
    for index, path in enumerate(paths):
        audio = decode_audio(str(ROOT / path), sampling_rate=16000)
        texts.append(transcribe_audio(model, audio)["text"])
        if (index + 1) % 100 == 0:
            print(f"  transcribed {index + 1}/{len(paths)}", flush=True)
    return texts


def transcribe_app_parallel(model_dir, paths, workers=3, threads=3):
    """Same as transcribe_app, split over several processes (results keep the input order)."""
    from concurrent.futures import ProcessPoolExecutor

    chunks = [list(paths[i::workers]) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(transcribe_app, [model_dir] * workers, chunks, [threads] * workers))
    texts = [None] * len(paths)
    for i, part in enumerate(parts):
        texts[i::workers] = part
    return texts


def transcribe_hf(model, paths, device, batch_size=16):
    """Greedy, no timestamps; ``model`` is a (possibly LoRA-wrapped) WhisperForConditionalGeneration.

    Runs in float32: on Apple MPS (torch 2.8) half-precision SDPA attention corrupts
    Whisper's outputs (loss 0.58 → 1.28 on the same batch), see PIPELINE.md.
    """
    import torch

    from .dataset import features, load_audio, processor

    tokenizer = processor().tokenizer
    texts = []
    model.eval()
    for start in range(0, len(paths), batch_size):
        batch = np.stack([features(load_audio(p)) for p in paths[start:start + batch_size]])
        with torch.inference_mode():
            generated = model.generate(input_features=torch.from_numpy(batch).to(device),
                                       max_new_tokens=220, num_beams=1, do_sample=False)
        texts.extend(t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True))
    return texts


# ----------------------------------------------------------------------------- scoring
def score(frame, hypotheses, kind="whisper"):
    """Pairs: two tracks of one mixture, best speaker assignment. Singles: one utterance each."""
    frame = frame.assign(hypothesis=list(hypotheses))
    details, pair_rows, single_rows = [], [], []
    for recipe_id, group in frame.groupby("recipe_id", sort=True):
        if group["kind"].iloc[0] == "pair":
            counts = pair_counts(list(group.text_ls), list(group.hypothesis), kind)
            pair_rows.append(counts)
        else:
            counts = add_counts([edit_counts(group.text_ls.iloc[0], group.hypothesis.iloc[0], kind)])
            single_rows.append(counts)
        details.append({"recipe_id": recipe_id, "kind": group["kind"].iloc[0], **counts})
    result = {}
    if pair_rows:
        result["pairs"] = {**add_counts(pair_rows), "mixtures": len(pair_rows)}
    if single_rows:
        result["singles"] = {**add_counts(single_rows), "utterances": len(single_rows)}
    return result, pd.DataFrame(details)


def bootstrap_difference(details_a, details_b, iterations=2000, seed=0):
    """95% interval of WER(b) - WER(a) resampling mixtures (paired, same mixtures)."""
    merged = details_a.merge(details_b, on="recipe_id", suffixes=("_a", "_b"))
    errors_a, errors_b = merged.errors_a.to_numpy(), merged.errors_b.to_numpy()
    words = merged.reference_words_a.to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iterations):
        index = rng.integers(0, len(merged), len(merged))
        diffs.append((errors_b[index].sum() - errors_a[index].sum()) / words[index].sum())
    observed = (errors_b.sum() - errors_a.sum()) / words.sum()
    return {"difference": float(observed), "low": float(np.percentile(diffs, 2.5)),
            "high": float(np.percentile(diffs, 97.5)), "mixtures": int(len(merged))}


def run(set_name, model, engine, threads, label, workers=1):
    if set_name in ("test", "v4test"):
        raise SystemExit("Test sets are scored only once, through ml_pipeline_english/04_eval.py (Gate C).")
    frame = eval_tracks(set_name)
    started = time.monotonic()
    if engine == "app":
        model_dir = BASE_WHISPER_CT2 if model == "baseline" else ROOT / model
        paths = list(frame.path)
        hypotheses = (transcribe_app_parallel(model_dir, paths, workers, threads) if workers > 1
                      else transcribe_app(model_dir, paths, threads))
    else:
        raise SystemExit("Use the training loop for engine=hf checks.")
    result = {kind: score(frame, hypotheses, kind)[0] for kind in ("whisper", "legacy")}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{set_name}-{label}-{engine}.json"
    out.write_text(json.dumps({"set": set_name, "model": str(model), "engine": engine,
                               "seconds": round(time.monotonic() - started, 1), "metrics": result,
                               "tracks": frame.assign(hypothesis=hypotheses)[["track_id", "recipe_id", "kind", "text_ls", "hypothesis"]].to_dict("records")},
                              indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=("val", "v4val"), required=True)
    parser.add_argument("--model", default="baseline")
    parser.add_argument("--label", default=None)
    parser.add_argument("--engine", choices=("app",), default="app")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1, help="parallel recognizer processes")
    args = parser.parse_args()
    run(args.set, args.model, args.engine, args.threads,
        args.label or ("baseline" if args.model == "baseline" else "finetuned"), args.workers)


if __name__ == "__main__":
    main()
