"""Exercise the deployed app with a public noisy demo. Not a quality benchmark.

Run the local app first, then: .venv/bin/python scripts/smoke_noise.py
The fixed model and decoding settings are never changed by this check.
"""
import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main():
    clean, rate = sf.read(ROOT / "samples/two_voices_kk_ru_en/kazakh_and_russian.wav", dtype="float32")
    if clean.ndim != 1 or not np.isfinite(clean).all():
        raise ValueError("Expected finite mono demonstration audio")
    noise = np.random.default_rng(20261110).standard_normal(len(clean))
    noise *= np.sqrt(np.mean(clean ** 2)) / np.sqrt(np.mean(noise ** 2)) / 10 ** (10 / 20)
    noisy = clean + noise
    noisy *= min(1.0, 0.95 / np.max(np.abs(noisy)))
    stream = io.BytesIO()
    sf.write(stream, noisy, rate, format="WAV", subtype="PCM_16")
    started = time.monotonic()
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=60) as client:
        client.get("/api/session").raise_for_status()
        response = client.post("/api/jobs", data={"mode": "conversation", "two_voices": "true", "language": "auto"},
                               files={"file": ("public-demo-white-10db.wav", stream.getvalue(), "audio/wav")})
        response.raise_for_status()
        job_id = response.json()["id"]
        while time.monotonic() - started < 600:
            response = client.get(f"/api/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()
            if job["status"] == "error":
                raise RuntimeError(job.get("error", "Processing failed"))
            if job["status"] == "done":
                break
            time.sleep(2)
        else:
            raise TimeoutError("Demonstration exceeded 10 minutes")
        result = job["result"]
        sources = result.get("sources", [])
        if len(sources) != 2:
            raise RuntimeError(f"Expected two playable sources, got {len(sources)}")
        for source in sources:
            audio_response = client.get(source["audio_url"])
            audio_response.raise_for_status()
            track, track_rate = sf.read(io.BytesIO(audio_response.content))
            if track_rate != 16000 or not np.isfinite(track).all() or not len(track):
                raise RuntimeError("Invalid separated audio")
    report = {"date": "2026-11-10", "scope": "Functional smoke check, not a benchmark or model selection",
              "sample": "samples/two_voices_kk_ru_en/kazakh_and_russian.wav",
              "condition": "Two foreground voices plus generated white noise at 10 dB full-recording SNR",
              "seconds": round(time.monotonic() - started, 2), "source_tracks": len(sources),
              "transcripts": [source.get("text", "") for source in sources],
              "warnings": result.get("warnings", []), "weights_changed": False}
    (ROOT / "docs/NOISE_SMOKE_2026-11-10.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
