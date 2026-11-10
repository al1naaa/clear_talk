"""Noise, room response and two-speaker mixing at 16 kHz.

The noise and reverb recipes follow the earlier ClearTalk v3/v4 generators so the new
training data covers the same conditions as the existing validation and test mixtures.
"""

import json
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from .paths import NOISE_MANIFEST, ROOT

RATE = 16000
SPEECH_RMS = 0.07


def synthetic_noise(length, rng, kind):
    noise = rng.standard_normal(length)
    spectrum = np.fft.rfft(noise)
    frequencies = np.fft.rfftfreq(length, d=1 / RATE)
    if kind in ("pink", "brown"):
        spectrum /= np.maximum(frequencies, RATE / length) ** (0.5 if kind == "pink" else 1.0)
    elif kind == "band":
        spectrum *= (frequencies >= 250) & (frequencies <= 1800)
    elif kind != "white":
        raise ValueError(f"Unknown noise kind {kind}")
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, n=length)
    # Slow loudness changes, so the noise is not perfectly stationary.
    noise *= np.interp(np.linspace(0, 7, length), np.arange(8), rng.uniform(0.5, 1.5, 8))
    return (noise / max(float(np.sqrt(np.mean(noise ** 2))), 1e-8)).astype(np.float32)


@lru_cache(maxsize=1)
def noise_clips():
    return json.loads(NOISE_MANIFEST.read_text())["clips"]


@lru_cache(maxsize=512)
def _noise_audio(relative_path):
    audio, rate = sf.read(ROOT / relative_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return resample_poly(audio, RATE, rate).astype(np.float32)


def environmental_noise(relative_path, length, rng):
    """Random crop of an ESC-10 recording (looped if short); silent crops use the loudest window."""
    signal = _noise_audio(relative_path)
    if len(signal) < length:
        signal = np.tile(signal, 1 + int(np.ceil(length / len(signal))))
    start = int(rng.integers(0, len(signal) - length + 1))
    clip = signal[start:start + length].astype(np.float64)
    clip -= clip.mean()
    if np.sqrt(np.mean(clip ** 2)) < 1e-6:
        energy = np.convolve(signal.astype(np.float64) ** 2, np.ones(length), mode="valid")
        start = int(np.argmax(energy))
        clip = signal[start:start + length].astype(np.float64)
        clip -= clip.mean()
    rms = float(np.sqrt(np.mean(clip ** 2)))
    if rms < 1e-6:
        raise ValueError(f"Silent environmental recording: {relative_path}")
    return (clip / rms).astype(np.float32)


def mild_reverb(signal, rng):
    """Short synthetic room response; not a measured real room."""
    length = round(float(rng.uniform(0.08, 0.22)) * RATE)
    response = rng.standard_normal(length) * np.exp(-6 * np.arange(length) / length)
    response *= 0.04 / max(float(np.sqrt(np.sum(response ** 2))), 1e-8)
    response[0] = 1.0
    for _ in range(4):
        delay = int(rng.integers(round(0.006 * RATE), min(length, round(0.065 * RATE))))
        response[delay] += float(rng.uniform(-0.16, 0.16))
    return fftconvolve(signal, response, mode="full")[:len(signal)].astype(np.float32)


def load_speech(relative_path):
    audio, rate = sf.read(ROOT / relative_path, dtype="float32")
    if audio.ndim != 1 or rate != RATE:
        raise ValueError(f"Expected mono 16 kHz LibriSpeech audio: {relative_path}")
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-5:
        raise ValueError(f"Near-silent utterance: {relative_path}")
    return audio * (SPEECH_RMS / rms)


def speech_span(signal, threshold_db=-35.0, frame=320):
    """First and last 20 ms frame within ``threshold_db`` of the loudest frame, in seconds."""
    frames = len(signal) // frame
    if frames == 0:
        return 0.0, len(signal) / RATE
    energy = (signal[:frames * frame].reshape(frames, frame) ** 2).mean(axis=1)
    active = np.flatnonzero(10 * np.log10(energy + 1e-12) > 10 * np.log10(energy.max() + 1e-12) + threshold_db)
    return float(active[0] * frame / RATE), float((active[-1] + 1) * frame / RATE)


def to_pcm16(signal):
    """Peak-limit to 0.95 and quantize like a saved WAV upload."""
    scale = min(1.0, 0.95 / max(float(np.max(np.abs(signal))), 1e-8))
    return np.round(signal * scale * 32767).clip(-32768, 32767).astype(np.int16), scale


def render(recipe):
    """Render one recipe to (mixture16k_int16, clean sources [T, n] float32 on the same scale, spans)."""
    rng = np.random.default_rng(recipe["seed"])
    sources = [load_speech(s["path"]) for s in recipe["sources"]]
    if recipe.get("reverb"):
        sources = [mild_reverb(s, rng) for s in sources]
    for index, source in enumerate(sources):
        sources[index] = source * 10 ** (float(recipe["gains_db"][index]) / 20)
    offsets = [round(float(o) * RATE) for o in recipe["offsets_seconds"]]
    length = max(o + len(s) for o, s in zip(offsets, sources))
    placed = np.zeros((length, len(sources)), dtype=np.float32)
    spans = []
    for index, (offset, source) in enumerate(zip(offsets, sources)):
        placed[offset:offset + len(source), index] = source
        start, end = speech_span(source)
        spans.append([round(offset / RATE + start, 2), round(offset / RATE + end, 2)])
    clean = placed.sum(axis=1)
    if recipe["noise"] == "none":
        noise = np.zeros(length, dtype=np.float32)
    elif recipe["noise"] == "esc10":
        noise = environmental_noise(recipe["noise_path"], length, rng)
    else:
        noise = synthetic_noise(length, rng, recipe["noise"])
    if recipe["noise"] != "none":
        noise *= float(np.sqrt(np.mean(clean ** 2))) / 10 ** (float(recipe["snr_db"]) / 20)
    pcm, scale = to_pcm16(clean + noise)
    return pcm, placed * scale, spans
