"""Background noise for training (round 4): white and pink noise, and babble made of other people's speech.

Babble talkers are TRAINING-role utterances, never validation or test speech, and never the talkers that the
robustness check (cleartalk_ml.robustness) uses for its validation noise - otherwise the check would hear voices
the model was trained to ignore. Every noisy example is reproducible: its noise is seeded by the example's id.
"""

import hashlib

import numpy as np

from .paths import SAMPLE_RATE

KINDS = {"white": 0.2, "pink": 0.2, "babble": 0.6}     # share of the noisy examples
SNR_DB = {"white": (0.0, 20.0), "pink": (0.0, 20.0), "babble": (3.0, 20.0)}
TALKERS = (3, 5)                                       # people talking at once in babble


def talker_pool(count=3000, seed=2026101540) -> list[str]:
    """Paths of training utterances usable as babble, minus the robustness check's own talkers."""
    import json

    from .corpora import load_manifest
    from .robustness import PLAN
    from .splits import cleaned

    table = cleaned(load_manifest())
    table = table[(table.role == "train") & table.keep & table.duration.between(3, 20)].sort_values("utt_id")
    reserved = set(json.loads(PLAN.read_text(encoding="utf-8"))["babble_talkers"]) if PLAN.is_file() else set()
    table = table[~table.path.isin(reserved)]
    picks = np.sort(np.random.default_rng(seed).permutation(len(table))[:count])
    return table.path.iloc[picks].tolist()


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def spec_for(example_id: str, probability: float):
    """The noise of one training example (kind, SNR, seed), or None for a clean example."""
    rng = _rng("noise:" + example_id)
    if probability <= 0 or rng.random() >= probability:
        return None
    kind = str(rng.choice(list(KINDS), p=list(KINDS.values())))
    low, high = SNR_DB[kind]
    return {"kind": kind, "snr": float(rng.uniform(low, high)), "seed": int(rng.integers(0, 2 ** 31 - 1))}


def _pink(length, rng):
    white = rng.standard_normal(length)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(length, 1 / SAMPLE_RATE)
    spectrum[1:] /= np.sqrt(frequencies[1:])
    spectrum[0] = 0
    return np.fft.irfft(spectrum, n=length)


def _babble(length, rng, talkers):
    from .corpora import read_audio

    mix = np.zeros(length, dtype=np.float32)
    for path in rng.choice(talkers, size=int(rng.integers(TALKERS[0], TALKERS[1] + 1)), replace=False):
        voice = read_audio(path).astype(np.float32)
        voice = voice / (float(np.sqrt(np.mean(voice ** 2))) or 1.0)
        voice = np.tile(voice, int(np.ceil(length / max(len(voice), 1))) + 1)
        offset = int(rng.integers(0, max(1, len(voice) - length)))
        mix += voice[offset:offset + length]
    return mix


def add(audio, spec, talkers) -> np.ndarray:
    """``audio`` with the noise of ``spec`` at its SNR (measured against the RMS of the whole clean audio)."""
    if spec is None or len(audio) == 0:
        return audio
    rng = np.random.default_rng(spec["seed"])
    if spec["kind"] == "white":
        noise = rng.standard_normal(len(audio))
    elif spec["kind"] == "pink":
        noise = _pink(len(audio), rng)
    else:
        noise = _babble(len(audio), rng, talkers)
    speech = float(np.sqrt(np.mean(audio ** 2))) or 1e-4
    noise = noise.astype(np.float32) * (speech / (float(np.sqrt(np.mean(noise ** 2))) or 1.0) / (10 ** (spec["snr"] / 20)))
    out = audio.astype(np.float32) + noise
    peak = float(np.max(np.abs(out)))
    return out * (0.95 / peak) if peak > 0.95 else out
