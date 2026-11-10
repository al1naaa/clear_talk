"""API/input tests use a small deterministic processor; inference is tested separately."""

import pytest

pytest.importorskip("fastapi")  # app tests run in the app environment (.venv)

import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.audio import AudioValidationError, decode_limited
from backend.config import Settings


def wav_bytes(seconds=1.0, silence=False):
    samples = np.zeros(int(16000 * seconds), dtype=np.float32)
    if not silence:
        samples = (0.1 * np.sin(2 * np.pi * 220 * np.arange(len(samples)) / 16000)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16000, format="WAV")
    return buffer.getvalue()


class FakeProcessor:
    def process(self, job_id, job_dir, mode, update):
        update("Test processing", 50)
        return {
            "duration_seconds": 1.0,
            "processing_seconds": 0.01,
            "mode": mode,
            "original": {"audio_url": f"/media/{job_id}/original.wav", "text": "Test transcript.", "segments": [{"start": 0, "end": 1, "text": "Test transcript."}]},
            "sources": [],
            "warnings": ["Test fixture, not an inferred transcript."],
            "models": {"asr": "test fixture", "separator": None},
            "download_url": f"/api/jobs/{job_id}/download",
        }


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(root=self.root, max_duration_seconds=2)
        self.settings.recognizer_dir("baseline").mkdir(parents=True)
        self.settings.separator_dir.mkdir(parents=True)
        for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
            (self.settings.recognizer_dir("baseline") / name).touch()
        for name in ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"):
            (self.settings.separator_dir / name).touch()
        self.settings.demo_dir.mkdir(parents=True)
        (self.settings.demo_dir / "mix.wav").write_bytes(wav_bytes())
        (self.settings.demo_dir / "metadata.json").write_text(json.dumps({"title": "Test demo"}))
        self.app = create_app(self.settings, FakeProcessor())
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.app.state.jobs.executor.shutdown(wait=True)
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def wait_done(self, job_id):
        for _ in range(100):
            payload = self.client.get(f"/api/jobs/{job_id}").json()
            if payload["status"] in {"done", "error"}:
                return payload
            time.sleep(0.01)
        self.fail("Test job did not finish.")

    def test_health_and_demo(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ready")
        self.assertEqual(self.client.get("/api/demo").json()["title"], "Test demo")
        self.assertEqual(self.client.get("/media/demo/mix.wav").status_code, 200)

    def test_demo_runs_and_download_has_actual_result(self):
        response = self.client.post("/api/jobs", data={"demo": "true", "mode": "transcribe"})
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["id"]
        job = self.wait_done(job_id)
        self.assertEqual(job["status"], "done")
        self.assertIn("Test transcript.", self.client.get(job["result"]["download_url"]).text)
        self.assertEqual(self.client.get(job["result"]["original"]["audio_url"]).status_code, 200)
        self.assertEqual(self.client.get(f"/media/{job_id}/result.json").status_code, 404)

    def test_upload_is_decoded_and_raw_input_removed(self):
        response = self.client.post("/api/jobs", files={"file": ("test.wav", wav_bytes(), "audio/wav")})
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["id"]
        self.assertEqual(self.wait_done(job_id)["status"], "done")
        self.assertFalse((self.settings.jobs_dir / job_id / "upload.wav").exists())

    def test_too_long_clip_is_rejected_and_cleaned(self):
        response = self.client.post("/api/jobs", files={"file": ("long.wav", wav_bytes(2.1), "audio/wav")})
        self.assertEqual(response.status_code, 422)
        self.assertIn("2 seconds", response.json()["detail"])
        self.assertEqual(list(self.settings.jobs_dir.iterdir()), [])

    def test_silent_and_invalid_files_are_rejected(self):
        for data in (wav_bytes(silence=True), b"not audio"):
            response = self.client.post("/api/jobs", files={"file": ("test.wav", data, "audio/wav")})
            self.assertEqual(response.status_code, 422)

    def test_missing_models_and_invalid_mode(self):
        self.assertEqual(self.client.post("/api/jobs", data={"demo": "true", "mode": "invalid"}).status_code, 422)
        (self.settings.separator_dir / "masknet.ckpt").unlink()
        self.assertEqual(self.client.post("/api/jobs", data={"demo": "true"}).status_code, 503)
        self.assertEqual(self.client.get("/api/health").json()["status"], "setup_required")

    def test_ambiguous_input_is_rejected(self):
        response = self.client.post("/api/jobs", data={"demo": "true"}, files={"file": ("x.wav", wav_bytes())})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post("/api/jobs").status_code, 422)


if __name__ == "__main__":
    unittest.main()


def test_page_assets_are_versioned_and_revalidated(tmp_path):
    """A browser must never pair a fresh page with a stale cached script."""
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text('<script src="/static/app.js" defer></script><link href="/static/styles.css">')
    (web / "app.js").write_text("// fixture")
    (web / "styles.css").write_text("/* fixture */")
    client = TestClient(create_app(Settings(root=tmp_path), FakeProcessor()))
    page = client.get("/")
    assert page.headers["cache-control"] == "no-cache"
    stamp = int((web / "app.js").stat().st_mtime)
    assert f"/static/app.js?v={stamp}" in page.text and "/static/styles.css?v=" in page.text
    script = client.get(f"/static/app.js?v={stamp}")
    assert script.status_code == 200 and script.headers["cache-control"] == "no-cache"
