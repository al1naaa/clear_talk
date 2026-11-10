"""Build a reproducible, clearly labelled two-speaker mixture from CC-BY audio."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf
from faster_whisper.audio import decode_audio

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
DOWNLOADS = ROOT / "data" / "downloads"
RATE = 16000
SAMPLES = [
    {
        "original_id": "5703-47212-0000",
        "sha256": "a284612b46af0535f7e1873758c4387bb8369f6dbbe192ffdec1f171108f98dd",
        "attribution": "Ashiel Mystery, chapter 2, narrated by Garth Comira. LibriSpeech SLR12; hosted by librosa.",
    },
    {
        "original_id": "198-209-0000",
        "sha256": "0473ba9613befa1026db5e254546f105aeffba8366b5468fc5e8d474d844ca1d",
        "attribution": "Sense and Sensibility, chapter 18, Jane Austen, read by Heather Barnett. LibriSpeech SLR12; hosted by librosa.",
    },
]


def main():
    DEMO.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    signals = []
    references = []
    for i, sample in enumerate(SAMPLES):
        name = sample["original_id"] + ".ogg"
        url = "https://librosa.org/data/audio/" + name
        path = DOWNLOADS / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            with urlopen(url, timeout=60) as response:
                content = response.read(5 * 1024 * 1024)
            if hashlib.sha256(content).hexdigest() != sample["sha256"]:
                raise RuntimeError(f"Audio checksum failed: {name}")
            path.write_bytes(content)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
            raise RuntimeError(f"Existing audio checksum failed: {name}")
        audio = decode_audio(str(path), sampling_rate=RATE)
        audio = audio[: RATE * 18].astype(np.float64)
        audio -= audio.mean()
        audio *= 0.09 / max(float(np.sqrt(np.mean(audio**2))), 1e-8)
        if i:
            audio = np.pad(audio, (int(0.4 * RATE), 0))
        signals.append(audio)
        references.append({
            "label": f"Reference {i + 1}",
            "file": f"source{i + 1}.wav",
            "original_id": sample["original_id"],
            "attribution": sample["attribution"],
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_url": url,
            "sha256": sample["sha256"],
        })
    length = max(map(len, signals))
    signals = [np.pad(x, (0, length - len(x))) for x in signals]
    clean_mix = signals[0] + signals[1]
    rng = np.random.default_rng(20261008)
    noise = rng.standard_normal(length)
    noise *= np.sqrt(np.mean(clean_mix**2) / 10) / np.sqrt(np.mean(noise**2))
    mix = clean_mix + noise
    scale = min(1.0, 0.94 / max(float(np.max(np.abs(mix))), 1e-8))
    for i, signal in enumerate(signals):
        sf.write(DEMO / f"source{i + 1}.wav", signal * scale, RATE, subtype="FLOAT")
    sf.write(DEMO / "noise.wav", noise * scale, RATE, subtype="FLOAT")
    sf.write(DEMO / "mix.wav", mix * scale, RATE, subtype="FLOAT")
    metadata = {
        "title": "Two voices, one noisy recording",
        "description": "Two public LibriSpeech audiobook excerpts mixed with generated white noise at 10 dB SNR. A controlled demonstration, not a real meeting or benchmark result.",
        "sample_rate": RATE,
        "duration_seconds": length / RATE,
        "snr_db": 10,
        "synthetic_mixture": True,
        "seed": 20261008,
        "reference_sources": references,
        "transcript_reference_available": False,
        "changes": "Resampled to 16 kHz, RMS-normalized, second source delayed 0.4 s, padded, mixed and scaled together. Generated Gaussian noise added; originals preserved in data/downloads.",
    }
    (DEMO / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Demo prepared: {length / RATE:.2f} seconds. Attribution saved in data/demo/metadata.json.")


if __name__ == "__main__":
    main()
