"""Adapt SepFormer's synthesis decoder on constructed KK/RU/EN foreground pairs.

The analysis encoder and mask estimator stay frozen. Choose the synthesis decoder
on development SI-SDR only, then check transcript errors against both frozen systems.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from speechbrain.inference.separation import SepformerSeparation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_three_language_asr import LANGS, contaminate, rank, split_rows, wave
from cleartalk_asr.augment import synthetic_noise

PAIRS = (("kk", "ru"), ("ru", "en"), ("en", "kk"))
N = 24000
EPOCHS = (1, 3, 6)


def crop(row, rng):
    x = resample_poly(wave(row), 1, 2)
    # Include short clips by zero padding, while avoiding silent-only inputs.
    start = int(rng.integers(max(1, len(x) - N + 1)))
    cut = x[start : start + N]
    if np.mean(cut * cut) < 1e-6:
        start = max(0, min(len(x) - N, int(np.argmax(abs(x))) - N // 2))
        cut = x[start : start + N]
    out = np.pad(cut, (0, N - len(cut)))
    return out * (0.07 / max(np.sqrt(np.mean(out * out)), 1e-6))


def make_examples(role):
    n = 64 if role == "train" else 8
    pools = split_rows(role, 2 * n, "sep-acoustic-" + role)
    pos = {l: 0 for l in LANGS}
    rows = []
    for a, b in PAIRS:
        for i in range(n):
            rng = np.random.default_rng(int(rank(f"{a}-{b}-{i}", role)[:16], 16))
            x = crop(pools[a][pos[a]], rng)
            y = crop(pools[b][pos[b]], rng)
            pos[a] += 1
            pos[b] += 1
            offset = int(rng.integers(0, 2000))
            if offset:
                y = np.pad(y[:-offset], (offset, 0))
            y *= 10 ** (float(rng.uniform(-5, 5)) / 20)
            ref = np.stack((x, y)).astype("float32")
            mix = ref.sum(axis=0)
            condition = "clean" if i % 4 == 0 else "noise"
            if condition == "noise":
                noise = synthetic_noise(N, rng, "white" if i % 2 else "pink")
                mix = contaminate(mix, noise, float(rng.uniform(3, 15)))
            rows.append((mix, ref, condition))
    return rows


def sisdr(est, target):
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (
        (est * target).sum(dim=-1, keepdim=True)
        / target.square().sum(dim=-1, keepdim=True).clamp_min(1e-10)
        * target
    )
    return 10 * torch.log10(
        projection.square().sum(-1).clamp_min(1e-10)
        / (est - projection).square().sum(-1).clamp_min(1e-10)
    )


def pit(est, ref):
    return torch.maximum(sisdr(est, ref).mean(1), sisdr(est, ref.flip(1)).mean(1))


def make_cache(model, device, examples, path):
    path.mkdir(parents=True, exist_ok=True)
    for i, (mix, ref, condition) in enumerate(examples):
        file = path / f"{i:03d}.pt"
        if file.exists():
            continue
        with torch.inference_mode():
            audio = torch.from_numpy(mix).to(device)[None]
            enc = model.mods.encoder(audio)
            mask = model.mods.masknet(enc)
            latent = (mask * enc.unsqueeze(0))[:, 0]
        tmp = file.with_suffix(".tmp")
        torch.save(
            dict(
                latent=latent.cpu().half(),
                ref=torch.from_numpy(ref),
                condition=condition,
                mixture_score=sisdr(
                    torch.from_numpy(mix)[None].expand(2, -1), torch.from_numpy(ref)
                ).mean(),
            ),
            tmp,
        )
        tmp.replace(file)
        if (i + 1) % 16 == 0:
            print(path.name, "cached", i + 1, flush=True)
    return sorted(path.glob("*.pt"))


def batch(decoder, rows):
    latent = torch.cat([r["latent"] for r in rows]).float()
    ref = torch.stack([r["ref"] for r in rows])
    est = decoder(latent).reshape(len(rows), 2, -1)
    return pit(est, ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "models/pitch-separator")
    parser.add_argument("--device", default="mps")
    opt = parser.parse_args()
    opt.work.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261121)
    torch.set_num_threads(2)
    source = ROOT / "models/candidates-v3/sepformer-libri2mix"
    model = SepformerSeparation.from_hparams(
        source=str(source), savedir=str(source), run_opts={"device": opt.device}
    )
    model.mods.eval()
    cache = {
        role: make_cache(model, opt.device, make_examples(role), opt.work / role)
        for role in ("train", "val")
    }
    decoder = copy.deepcopy(model.mods.decoder).cpu().requires_grad_(True)
    del model
    if opt.device == "mps":
        torch.mps.empty_cache()
    initial = copy.deepcopy(decoder.state_dict())
    val = [torch.load(f, weights_only=True) for f in cache["val"]]

    def assess():
        with torch.no_grad():
            scores = torch.cat(
                [batch(decoder, val[i : i + 4]) for i in range(0, len(val), 4)]
            )
        gain = scores - torch.stack([r["mixture_score"] for r in val])
        return {
            c: float(
                gain[
                    [i for i, r in enumerate(val) if c == "all" or r["condition"] == c]
                ].mean()
            )
            for c in ("all", "clean", "noise")
        }

    history = {0: assess()}
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=0)
    for epoch in range(1, max(EPOCHS) + 1):
        order = np.random.default_rng(20261121 + epoch).permutation(len(cache["train"]))
        for idx in np.array_split(order, len(order) // 8):
            rows = [torch.load(cache["train"][int(i)], weights_only=True) for i in idx]
            optimizer.zero_grad(set_to_none=True)
            loss = -batch(decoder, rows).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), 2.0, error_if_nonfinite=True
            )
            optimizer.step()
        if epoch in EPOCHS:
            history[epoch] = assess()
            torch.save(decoder.state_dict(), opt.work / f"epoch-{epoch}.ckpt")
            print(epoch, history[epoch], flush=True)
    safe = [e for e, v in history.items() if v["clean"] >= history[0]["clean"] - 0.1]
    winner = max(safe, key=lambda e: history[e]["all"])
    if winner:
        decoder.load_state_dict(
            torch.load(opt.work / f"epoch-{winner}.ckpt", weights_only=True)
        )
    else:
        decoder.load_state_dict(initial)
    torch.save(decoder.state_dict(), opt.work / "selected.ckpt")
    out = dict(
        epoch=winner,
        train_pairs=len(cache["train"]),
        validation_pairs=len(val),
        history=history,
        trainable_parameters=sum(p.numel() for p in decoder.parameters()),
        criterion="Largest mean validation SI-SDR improvement, clean slice within 0.1 dB of the original decoder.",
    )
    (opt.work / "selection.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
