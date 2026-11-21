"""Fit a new noise-aware LoRA on a verified native step-600 starting model.

Fixed 96-step budget. Each update sees six utterances, two per language, split
equally between clean and corrupted. No validation or test signal is used to fit.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cleartalk_asr.augment import environmental_noise, noise_clips, synthetic_noise

LANGS = ("kk", "ru", "en")
SEED = 20261122
STEPS = 96
CHECKPOINTS = (24, 48, 96)


def rank(uid, salt):
    return hashlib.sha256(f"{SEED}/{salt}/{uid}".encode()).hexdigest()


def split_rows(role, per_language, salt):
    with (ROOT / "data/ml/noise_followup_20261110/development_manifest.csv").open(
        encoding="utf-8"
    ) as f:
        all_rows = list(csv.DictReader(f))
    pools = {}
    for l in LANGS:
        rows = [
            r
            for r in all_rows
            if r["split"] == role
            and r["language"] == l
            and not r["excluded_reason"]
            and 2 <= float(r["duration"]) <= 18
        ]
        rows.sort(key=lambda r: rank(r["utt_id"], salt))
        seen = set()
        pools[l] = []
        for r in rows:
            if r["group"] not in seen:
                pools[l].append(r)
                seen.add(r["group"])
            if len(pools[l]) == per_language:
                break
        assert len(pools[l]) == per_language
    return pools


def wave(row):
    x, rate = sf.read(ROOT / row["path"], dtype="float32")
    assert rate == 16000 and x.ndim == 1
    return x


def contaminate(x, noise, db):
    target = max(float(np.sqrt(np.mean(x * x))), 1e-8)
    rms = max(float(np.sqrt(np.mean(noise * noise))), 1e-8)
    out = x + noise * (target / rms / 10 ** (db / 20))
    return (out * min(1.0, 0.95 / max(float(abs(out).max()), 1e-8))).astype("float32")


def babble(length, pool, rng):
    total = np.zeros(length, dtype="float32")
    for i in rng.choice(len(pool), 4, replace=False):
        speech = wave(pool[int(i)])
        tiled = np.tile(speech, 2 + length // len(speech))
        pos = int(rng.integers(0, len(speech)))
        cut = tiled[pos : pos + length]
        total += cut / max(float(np.sqrt(np.mean(cut * cut))), 1e-7)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.work.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(3)
    base = ROOT / "models/training-round2"
    assert (base / "verified_sha256.txt").is_file()
    training = split_rows("train", STEPS * 2, "asr-fit")
    validation = split_rows("val", 12, "asr-val")
    train_ids = {r["utt_id"] for pool in training.values() for r in pool}
    validation_groups = {v["group"] for vs in validation.values() for v in vs}
    with (
        ROOT / "data/ml/noise_followup_20261110/development_manifest.csv"
    ).open() as f:
        background = [
            r
            for r in csv.DictReader(f)
            if r["split"] == "train"
            and r["utt_id"] not in train_ids
            and not r["excluded_reason"]
            and r["group"] not in validation_groups
            and 3 <= float(r["duration"]) <= 15
        ]
    background = sorted(background, key=lambda r: rank(r["utt_id"], "babble"))[:160]
    assert not (
        {r["group"] for rs in training.values() for r in rs}
        & {r["group"] for rs in validation.values() for r in rs}
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        base, dtype=torch.float32, attn_implementation="sdpa"
    )
    targets = [
        n
        for n, _ in model.named_modules()
        if n.endswith((".q_proj", ".v_proj"))
        and (
            n.startswith(("model.encoder.layers.30.", "model.encoder.layers.31."))
            or n.startswith("model.decoder.layers.")
            and ".encoder_attn." in n
        )
    ]
    assert len(targets) == 12
    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, target_modules=targets, bias="none"
        ),
    )
    model.to(args.device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=6e-5, weight_decay=0.01)
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    extractor = WhisperFeatureExtractor.from_pretrained(base)
    source_clips = [c["path"] for c in noise_clips() if c["group"] == "train"]
    config = dict(
        seed=SEED,
        steps=STEPS,
        examples=6 * STEPS,
        trainable=sum(p.numel() for p in params),
        checkpoints=CHECKPOINTS,
        device=args.device,
        targets=targets,
        base_sha=(base / "verified_sha256.txt").read_text().strip(),
        lr=6e-5,
        rank=8,
        alpha=16,
        dropout=0.05,
        validation_rule="Lowest macro noisy validation WER among clean-safe candidates; keep incumbent on ties or failure.",
    )
    (args.work / "config.json").write_text(json.dumps(config, indent=2))
    latest = args.work / "resume.pt"
    start = 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        if "cpu_rng" in state:
            torch.set_rng_state(state["cpu_rng"])
        if args.device == "mps" and state.get("device_rng") is not None:
            torch.mps.set_rng_state(state["device_rng"])
        elif args.device.startswith("cuda") and state.get("device_rng") is not None:
            torch.cuda.set_rng_state(state["device_rng"], device=args.device)
    print(config, flush=True)
    for step in range(start, STEPS):
        begun = time.monotonic()
        avg_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        learning_rate = (
            6e-5
            * min(1.0, (step + 1) / 8)
            * (0.25 + 0.75 * (1 + math.cos(math.pi * step / STEPS)) / 2)
        )
        for g in optimizer.param_groups:
            g["lr"] = learning_rate
        for view in range(2):
            audio = []
            labels = []
            for j, lang in enumerate(LANGS):
                row = training[lang][2 * step + view]
                x = wave(row)
                if (step + view + j) % 2:
                    rng = np.random.default_rng(SEED + step * 6 + view * 3 + j)
                    kind = ("white", "pink", "esc", "babble")[(step + j) % 4]
                    n = (
                        environmental_noise(
                            source_clips[int(rng.integers(len(source_clips)))],
                            len(x),
                            rng,
                        )
                        if kind == "esc"
                        else babble(len(x), background, rng)
                        if kind == "babble"
                        else synthetic_noise(len(x), rng, kind)
                    )
                    x = contaminate(x, n, float(rng.uniform(3, 18)))
                audio.append(x)
                prefix = [
                    tokenizer.token_to_id(t)
                    for t in (f"<|{lang}|>", "<|transcribe|>", "<|notimestamps|>")
                ]
                labels.append(
                    prefix
                    + tokenizer.encode(
                        row["text_raw"].strip(), add_special_tokens=False
                    ).ids
                    + [tokenizer.token_to_id("<|endoftext|>")]
                )
            for signal, ids in zip(audio, labels, strict=True):
                features = extractor(
                    signal, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(args.device)
                if len(ids) > 192:
                    raise ValueError("Unexpected transcript length")
                batch = torch.full((1, 192), -100, dtype=torch.long)
                batch[0, : len(ids)] = torch.tensor(ids)
                loss = model(
                    input_features=features,
                    labels=batch.to(args.device),
                    use_cache=False,
                ).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                (loss / 6).backward()
                avg_loss += float(loss.detach()) / 6
                del features, batch, loss
                if args.device == "mps":
                    torch.mps.empty_cache()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        print(
            f"step {step + 1}/{STEPS} loss {avg_loss:.4f} seconds {time.monotonic() - begun:.1f}",
            flush=True,
        )
        if step + 1 in CHECKPOINTS:
            model.save_pretrained(
                args.work / f"step-{step + 1:03d}", save_embedding_layers=False
            )
            from peft import get_peft_model_state_dict

            torch.save(
                dict(
                    step=step + 1,
                    adapter=get_peft_model_state_dict(model),
                    optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(),
                    device_rng=(
                        torch.mps.get_rng_state()
                        if args.device == "mps"
                        else torch.cuda.get_rng_state(args.device)
                        if args.device.startswith("cuda")
                        else None
                    ),
                ),
                latest.with_suffix(".tmp"),
            )
            os.replace(latest.with_suffix(".tmp"), latest)


if __name__ == "__main__":
    main()
