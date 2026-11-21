"""Save a real completed local job as a self-contained offline listening demo."""

import argparse
import base64
import html
import io
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf


def audio_player(data):
    encoded = base64.b64encode(data).decode("ascii")
    return f'<audio controls src="data:audio/wav;base64,{encoded}"></audio>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--input-copy", type=Path, help="Also save the actual noisy WAV for live upload"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    clean, rate = sf.read(root / "samples/two_voices_kk_ru_en/kazakh_and_russian.wav")
    noise = np.random.default_rng(20261110).standard_normal(len(clean))
    signal = clean + noise * np.linalg.norm(clean) / (
        np.linalg.norm(noise) * np.sqrt(10)
    )
    stream = io.BytesIO()
    sf.write(stream, signal * min(1, 0.95 / abs(signal).max()), rate, format="WAV")
    mixture = stream.getvalue()
    if args.input_copy:
        args.input_copy.parent.mkdir(parents=True, exist_ok=True)
        args.input_copy.write_bytes(mixture)
    started = time.monotonic()
    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}", timeout=60, trust_env=False
    ) as client:
        client.get("/api/session").raise_for_status()
        reply = client.post(
            "/api/jobs",
            data={"mode": "conversation", "two_voices": "true", "language": "auto"},
            files={"file": ("pitch.wav", mixture, "audio/wav")},
        )
        reply.raise_for_status()
        job_url = "/api/jobs/" + reply.json()["id"]
        while time.monotonic() - started < 600:
            reply = client.get(job_url)
            reply.raise_for_status()
            job = reply.json()
            if job["status"] == "error":
                raise RuntimeError(job.get("error"))
            if job["status"] == "done":
                break
            time.sleep(1)
        else:
            raise TimeoutError("Local job exceeded ten minutes")
        elapsed = time.monotonic() - started
        sources = job["result"].get("sources", [])
        if len(sources) != 2:
            raise RuntimeError("Two separate outputs were not returned")
        parts = [
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>ClearTalk listening demo</title>',
            '<meta name="viewport" content="width=device-width, initial-scale=1"><style>body{max-width:760px;margin:6vh auto;padding:0 24px;font:19px/1.5 system-ui;color:#132c38;background:#f5faf9}h1{line-height:1.1}audio{width:100%}section{border-top:1px solid #aac2bb;margin-top:28px}</style>',
            "<h1>Two simultaneous voices</h1><p>Saved inference · Kazakh and Russian · white noise at 10 dB. Listen to the mixture, then compare the two outputs and their unedited transcripts.</p>",
            f"<p>{len(clean) / rate:.2f} seconds of audio · {elapsed:.1f} seconds from upload to result</p><h2>Input mixture</h2>",
            audio_player(mixture),
        ]
        for index, source in enumerate(sources, 1):
            reply = client.get(source["audio_url"])
            reply.raise_for_status()
            track, sr = sf.read(io.BytesIO(reply.content))
            if (
                sr != 16000
                or len(track) != len(clean)
                or not np.isfinite(track).all()
                or not source.get("text")
            ):
                raise ValueError("Invalid source audio or missing transcript")
            parts += [
                f"<section><h2>Speaker {index}</h2>",
                audio_player(reply.content),
                "<p>" + html.escape(source["text"]) + "</p></section>",
            ]
            print(index, source["text"], flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(parts) + "</html>\n", encoding="utf-8")
        print(f"{elapsed:.1f} seconds; saved two playable tracks")


if __name__ == "__main__":
    main()
