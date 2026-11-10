"""Service boundaries; these fixtures intentionally do not claim ML quality."""

import pytest

pytest.importorskip("fastapi")  # app tests run in the app environment (.venv)

import dataclasses
import asyncio
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from backend.security import ServiceMiddleware, SessionSecurity
from test_backend import FakeProcessor, wav_bytes


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(root=Path(self.temp.name), max_duration_seconds=2)
        self.settings.recognizer_dir("baseline").mkdir(parents=True)
        self.settings.separator_dir.mkdir(parents=True)
        for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
            (self.settings.recognizer_dir("baseline") / name).touch()
        for name in ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"):
            (self.settings.separator_dir / name).touch()
        self.settings.demo_dir.mkdir(parents=True)
        (self.settings.demo_dir / "mix.wav").write_bytes(wav_bytes())
        self.instances = []

    def tearDown(self):
        for app, client in self.instances:
            app.state.jobs.executor.shutdown(wait=True)
            client.__exit__(None, None, None)
        self.temp.cleanup()

    def client(self, settings=None, processor=None):
        app = create_app(settings or self.settings, processor or FakeProcessor())
        client = TestClient(app)
        client.__enter__()
        self.instances.append((app, client))
        return app, client

    def done(self, client, job_id):
        for _ in range(100):
            response = client.get(f"/api/jobs/{job_id}")
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            if result["status"] in {"done", "error"}:
                return result
            time.sleep(0.01)
        self.fail("Fixture job did not finish")

    def submit(self, client, model="baseline"):
        response = client.post("/api/jobs", data={"demo": "true", "mode": "transcribe", "model": model})
        self.assertEqual(response.status_code, 202, response.text)
        return self.done(client, response.json()["id"])

    def test_private_job_media_and_download_require_same_browser(self):
        app, client = self.client()
        response = client.get("/api/session")
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        job = self.submit(client)
        self.assertNotIn("owner", job)
        with TestClient(app) as stranger:
            for url in (f"/api/jobs/{job['id']}", job["result"]["download_url"], job["result"]["original"]["audio_url"]):
                self.assertEqual(stranger.get(url).status_code, 404)
            self.assertEqual(stranger.delete(f"/api/jobs/{job['id']}").status_code, 404)
        self.assertEqual(client.get(job["result"]["original"]["audio_url"]).status_code, 200)

    def test_restart_recovers_finished_job_for_its_owner(self):
        first, client = self.client()
        job = self.submit(client)
        first.state.jobs.executor.shutdown(wait=True)
        _, restored = self.client()
        restored.cookies.update(client.cookies)
        response = restored.get(f"/api/jobs/{job['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "done")
        self.assertIn("Test transcript.", restored.get(job["result"]["download_url"]).text)

    def test_interrupted_job_is_marked_failed_after_restart(self):
        first, client = self.client()
        client.get("/api/session")
        owner = client.cookies.get("cleartalk_session").split(".")[0]
        job_id = first.state.jobs.reserve(owner, "transcribe", "baseline")
        _, restored = self.client()
        restored.cookies.update(client.cookies)
        result = restored.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(result["error_code"], "worker_restarted")

    def test_expiration_removes_owned_files(self):
        app, client = self.client()
        job = self.submit(client)
        with app.state.jobs.lock:
            app.state.jobs.jobs[job["id"]]["expires_at"] = time.time() - 1
        response = client.get(f"/api/jobs/{job['id']}")
        self.assertEqual(response.status_code, 404)
        self.assertFalse((self.settings.jobs_dir / job["id"]).exists())

    def test_delete_removes_own_recording(self):
        _, client = self.client()
        job = self.submit(client)
        self.assertEqual(client.delete(f"/api/jobs/{job['id']}").status_code, 204)
        self.assertFalse((self.settings.jobs_dir / job["id"]).exists())

    def test_access_gate_is_required_for_upload_and_survives_restart(self):
        settings = dataclasses.replace(self.settings, access_token="long-deployment-secret")
        _, client = self.client(settings)
        self.assertEqual(client.get("/api/health").status_code, 200)
        self.assertFalse(client.get("/api/session").json()["authenticated"])
        self.assertEqual(client.post("/api/jobs", data={"demo": "true"}).status_code, 401)
        self.assertEqual(client.post("/api/access", json={"token": "incorrect"}).status_code, 401)
        self.assertEqual(client.post("/api/access", json={"token": settings.access_token}).status_code, 200)
        self.assertTrue(client.get("/api/session").json()["authenticated"])
        job = self.submit(client)
        _, restored = self.client(settings)
        restored.cookies.update(client.cookies)
        self.assertTrue(restored.get("/api/session").json()["authenticated"])
        client.cookies.delete("cleartalk_access")
        for url in (f"/api/jobs/{job['id']}", job["result"]["download_url"], job["result"]["original"]["audio_url"]):
            self.assertEqual(client.get(url).status_code, 401)

    def test_forged_session_signature_cannot_read_job(self):
        app, client = self.client()
        job = self.submit(client)
        cookie = client.cookies.get("cleartalk_session")
        with TestClient(app) as stranger:
            stranger.cookies.set("cleartalk_session", cookie.split(".")[0] + "." + "0" * 64)
            self.assertEqual(stranger.get(f"/api/jobs/{job['id']}").status_code, 404)

    def test_request_and_exact_file_limits_are_enforced(self):
        settings = dataclasses.replace(self.settings, max_upload_bytes=1024)
        _, client = self.client(settings)
        response = client.post("/api/jobs", files={"file": ("x.wav", wav_bytes())})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "upload_too_large")
        # No Content-Length: bytes must be counted while the body is consumed.
        boundary = "testBoundary"
        payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode() + b"x" * 70000 + f"\r\n--{boundary}--\r\n".encode())
        response = client.post("/api/jobs", content=iter([payload[:1000], payload[1000:]]), headers={"content-type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(list(settings.jobs_dir.iterdir()), [])

    def test_queue_and_storage_limits_reject_before_creating_recording(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingProcessor(FakeProcessor):
            def process(self, *args):
                entered.set()
                release.wait(timeout=5)
                return super().process(*args)

        app, client = self.client(dataclasses.replace(self.settings, max_pending_jobs=1), BlockingProcessor())
        try:
            response = client.post("/api/jobs", data={"demo": "true"})
            self.assertEqual(response.status_code, 202)
            self.assertTrue(entered.wait(timeout=2))
            self.assertEqual(client.post("/api/jobs", data={"demo": "true"}).status_code, 429)
            self.assertEqual(len(list(self.settings.jobs_dir.iterdir())), 1)
        finally:
            release.set()
        self.done(client, response.json()["id"])
        app.state.jobs.executor.shutdown(wait=True)
        _, full_client = self.client(dataclasses.replace(self.settings, max_stored_jobs=1))
        self.assertEqual(full_client.post("/api/jobs", data={"demo": "true"}).status_code, 507)

    def test_disk_budget_and_missing_cleartalk_model_do_not_run(self):
        _, client = self.client(dataclasses.replace(self.settings, max_storage_bytes=1000))
        self.assertEqual(client.post("/api/jobs", data={"demo": "true"}).status_code, 507)
        for mode in ("separate", "transcribe"):
            self.assertEqual(client.post("/api/jobs", data={"demo": "true", "model": "cleartalk", "mode": mode}).status_code, 409)
        self.assertEqual(client.post("/api/jobs", data={"demo": "true", "model": "custom"}).status_code, 422)
        self.assertEqual(list(self.settings.jobs_dir.iterdir()), [])

    def test_concurrent_uploads_are_bounded_before_multipart_parsing(self):
        async def check():
            entered, release = asyncio.Event(), asyncio.Event()

            async def slow_app(scope, receive, send):
                entered.set()
                await release.wait()
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"ok"})

            async def receive():
                self.fail("An over-capacity upload must be rejected before its body is read")

            first_messages, second_messages = [], []

            async def first_send(message):
                first_messages.append(message)

            async def second_send(message):
                second_messages.append(message)

            settings = dataclasses.replace(self.settings, max_pending_jobs=1)
            middleware = ServiceMiddleware(slow_app, settings, SessionSecurity(settings), SimpleNamespace(pending=lambda: 0))
            scope = {"type": "http", "method": "POST", "path": "/api/jobs", "headers": []}
            first = asyncio.create_task(middleware(dict(scope), receive, first_send))
            await asyncio.wait_for(entered.wait(), timeout=2)
            await middleware(dict(scope), receive, second_send)
            self.assertEqual(second_messages[0]["status"], 429)
            release.set()
            await first
            self.assertEqual(middleware.uploading, 0)

        asyncio.run(check())

    def test_installed_cleartalk_model_is_offered_with_its_card(self):
        directory = self.settings.recognizer_dir("cleartalk")
        directory.mkdir(parents=True)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (directory / name).write_bytes(b"fixture, never loaded")
        (directory / "model_card.json").write_text(json.dumps({"model_name": "fixture card"}))
        _, client = self.client()
        health = client.get("/api/health").json()
        self.assertTrue(health["models"]["cleartalk"]["available"])
        self.assertEqual(health["models"]["cleartalk"]["model_card"]["model_name"], "fixture card")
        self.assertEqual(self.submit(client, model="cleartalk")["status"], "done")

    def test_health_distinguishes_installed_files_from_inference(self):
        _, client = self.client()
        health = client.get("/api/health").json()
        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["readiness"], "not_loaded")
        self.assertFalse(health["inference_verified"])
        self.assertFalse(health["models"]["baseline"]["loaded"])
        self.assertFalse(health["features"]["visual_speech"])

    def test_nondict_model_cards_are_ignored_without_blocking_baseline(self):
        directory = self.settings.recognizer_dir("cleartalk")
        directory.mkdir(parents=True)
        card = directory / "model_card.json"
        _, client = self.client()
        for value in ([], None, 7, "invalid"):
            card.write_text(json.dumps(value))
            before = card.read_bytes()
            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertIsNone(health.json()["models"]["cleartalk"]["model_card"])
            self.assertEqual(client.post("/api/jobs", data={"demo": "true", "model": "cleartalk"}).status_code, 409)
            self.assertEqual(self.submit(client)["status"], "done")
            self.assertEqual(card.read_bytes(), before)

    def test_nondict_job_metadata_survives_startup_unchanged_and_unowned(self):
        snapshots = {}
        for index, value in enumerate(([], None, 7, "invalid"), 1):
            directory = self.settings.jobs_dir / f"{index:032x}"
            directory.mkdir(parents=True)
            (directory / "job.json").write_text(json.dumps(value))
            (directory / "original.wav").write_bytes(b"preserved fixture bytes, not audio")
            snapshots.update({path: path.read_bytes() for path in directory.iterdir()})
        app, client = self.client()
        self.assertEqual(client.get("/api/health").status_code, 200)
        self.assertEqual(app.state.jobs.jobs, {})
        for index in range(1, 5):
            self.assertEqual(client.get(f"/api/jobs/{index:032x}").status_code, 404)
        self.assertTrue(all(path.read_bytes() == value for path, value in snapshots.items()))


if __name__ == "__main__":
    unittest.main()
