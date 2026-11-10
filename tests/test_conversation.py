"""Conversation mode: language choice, speaker turns, routing and the service boundary.

Fixtures use fake recognizers and synthetic voice embeddings; they check logic, not ML quality.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from backend import conversation, speakers
from backend.overlap import Separator
from backend.app import create_app
from backend.config import DEFAULT_MULTILINGUAL_PROFILE, Settings, _safe_directory
from backend.jobs import transcript_text
from backend.multilingual import decoding_policy, restricted_language
from test_backend import FakeProcessor, wav_bytes


# ----------------------------------------------------------------------------- language choice
def test_detection_is_restricted_to_the_three_supported_languages():
    # Whisper's own first choice is Ukrainian here; among kk / ru / en the answer is Russian.
    language, share, shares = restricted_language([("uk", 0.5), ("ru", 0.3), ("kk", 0.1), ("en", 0.05)])
    assert language == "ru" and abs(share - 0.3 / 0.45) < 1e-9
    assert abs(sum(shares.values()) - 1) < 1e-9


def test_detection_without_scores_does_not_crash():
    assert restricted_language([])[0] in {"kk", "ru", "en"}


def test_policy_refuses_unsupported_languages():
    assert decoding_policy("kk")["language"] == "kk"
    with pytest.raises(ValueError):
        decoding_policy("de")


def test_short_turns_take_their_language_from_context():
    turn = lambda speaker, start, end: speakers.Turn(start, end, speaker)  # noqa: E731
    turns = [turn(0, 0, 5), turn(1, 5, 9), turn(0, 9, 9.8), turn(2, 10, 10.5)]
    found = lambda language: {"language": language, "uncertain": False}  # noqa: E731
    final = conversation.settle_languages(turns, [found("kk"), found("ru"), None, None])
    assert final == ["kk", "ru", "kk", "ru"]  # own speaker first, then the nearest trusted turn


def test_uncertain_detection_falls_back_to_its_own_guess_when_alone():
    turns = [speakers.Turn(0, 3, 0)]
    assert conversation.settle_languages(turns, [{"language": "en", "uncertain": True}]) == ["en"]


# ----------------------------------------------------------------------------- speaker turns
def synthetic_embeddings(counts, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.eye(len(counts), 32)
    rows = [centres[k] + 0.05 * rng.standard_normal(32) for k, count in enumerate(counts) for _ in range(count)]
    rows = np.array(rows)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_clustering_finds_the_number_of_speakers():
    labels = speakers.cluster(synthetic_embeddings([12, 9, 7]))
    assert len(set(labels.tolist())) == 3
    assert len(set(labels[:12].tolist())) == 1 and len(set(labels[12:21].tolist())) == 1


def test_clustering_respects_a_given_number_of_speakers():
    assert len(set(speakers.cluster(synthetic_embeddings([10, 10, 10]), speakers=2).tolist())) == 2
    assert set(speakers.cluster(synthetic_embeddings([5, 5]), speakers=1).tolist()) == {0}


def test_turns_are_numbered_by_first_appearance_and_merged():
    regions = [(0.0, 3.0), (3.4, 6.4), (10.0, 13.0)]
    windows = speakers.windows_of(regions)
    by_region = {0: 7, 1: 7, 2: 4}
    labels = [by_region[region] for region, _, _ in windows]
    turns = speakers.turns_from(regions, windows, labels)
    assert [(t.speaker, round(t.start, 1), round(t.end, 1)) for t in turns] == [(0, 0.0, 6.4), (1, 10.0, 13.0)]


def test_a_single_stray_window_is_smoothed_away():
    regions = [(0.0, 6.0)]
    windows = speakers.windows_of(regions)
    labels = [0] * len(windows)
    labels[3] = 1
    assert len(speakers.turns_from(regions, windows, labels)) == 1


# ----------------------------------------------------------------------------- routing
class FakeRecognizer:
    def __init__(self, name):
        self.name = name


def make_pipeline(tmp, monkeypatch, detections):
    settings = Settings(root=Path(tmp))
    pipeline = conversation.ConversationPipeline(settings)
    monkeypatch.setattr(pipeline, "recognizer", lambda key: FakeRecognizer(key))
    monkeypatch.setattr(pipeline, "device", lambda: ("cpu", "int8"))
    turns = [speakers.Turn(0.0, 4.0, 0), speakers.Turn(4.5, 9.0, 1), speakers.Turn(9.2, 9.9, 0)]
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: turns)
    queue = list(detections)
    monkeypatch.setattr(conversation, "detect_language", lambda model, audio: queue.pop(0))
    monkeypatch.setattr(conversation, "transcribe_window",
                        lambda model, audio, language: {"text": f"{model.name}:{language}", "language": language,
                                                        "segments": [{"needs_review": False}], "speech_seconds": 1.0})
    return pipeline


def test_each_language_goes_to_its_route(tmp_path, monkeypatch):
    found = lambda language: {"language": language, "uncertain": False}  # noqa: E731
    pipeline = make_pipeline(tmp_path, monkeypatch, [found("kk"), found("ru")])
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto")
    assert [t["text"] for t in result["turns"]] == ["kazakh:kk", "turbo:ru", "kazakh:kk"]
    assert [t["speaker"] for t in result["turns"]] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert result["turns"][2]["needs_review"]  # too short for its own language detection


def test_a_kazakh_only_recognizer_is_not_given_russian(tmp_path, monkeypatch):
    found = lambda language: {"language": language, "uncertain": False}  # noqa: E731
    pipeline = make_pipeline(tmp_path, monkeypatch, [found("kk"), found("ru")])
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="kazakh")
    assert [t["recognizer"] for t in result["turns"]] == ["kazakh", "turbo", "kazakh"]


def test_fixed_language_skips_detection(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [])
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="en", model="small")
    assert {t["text"] for t in result["turns"]} == {"small:en"}
    assert not any(t["needs_review"] for t in result["turns"])


# ----------------------------------------------------------------------------- profile and paths
@pytest.mark.parametrize("value", ["/absolute", "C:\\models", "..\\up", "../up", "\\\\server\\share", "", None, 3])
def test_unsafe_directories_are_rejected_on_every_system(value):
    assert not _safe_directory(value)


def test_safe_directories_are_accepted():
    assert _safe_directory("whisper-small") and _safe_directory("candidates-v3/whisper-small.en")


@pytest.mark.parametrize("change", [
    {"version": 2}, {"asr_policy": "unverified"}, {"detector": "missing"},
    {"routes": {"kk": "kazakh", "ru": "turbo"}},
    {"routes": {"kk": "kazakh", "ru": "kazakh", "en": "turbo"}},  # the Kazakh model does not cover Russian
    {"speaker_model": {"id": "x", "directory": "../outside"}},
])
def test_invalid_multilingual_profile_is_rejected(tmp_path, change):
    settings = Settings(root=tmp_path)
    settings.multilingual_dir.mkdir(parents=True)
    (settings.multilingual_dir / "profile.json").write_text(json.dumps({**DEFAULT_MULTILINGUAL_PROFILE, **change}))
    with pytest.raises(ValueError, match="Invalid multilingual"):
        settings.multilingual_profile


# ----------------------------------------------------------------------------- transcript export
def test_conversation_transcript_lists_turns_with_speaker_and_language():
    result = {"mode": "conversation", "duration_seconds": 3725.0, "models": {"asr": "fixture", "speakers": "fixture voices"},
              "turns": [{"start": 0.0, "end": 4.2, "speaker": "Speaker 1", "language_name": "Kazakh", "text": "Сәлем", "needs_review": False},
                        {"start": 3661.0, "end": 3665.0, "speaker": "Speaker 2", "language_name": "English", "text": "Hello", "needs_review": True}],
              "warnings": ["Fixture."]}
    text = transcript_text(result)
    assert "Audio duration: 1:02:05" in text
    assert "[0:00-0:04] Speaker 1 (Kazakh): Сәлем" in text
    assert "[1:01:01-1:01:05] Speaker 2 (English): Hello  [check]" in text


# ----------------------------------------------------------------------------- service boundary
class ConversationServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(root=Path(self.temp.name), max_duration_seconds=2, max_long_seconds=6)
        self.instances = []

    def tearDown(self):
        for app, client in self.instances:
            app.state.jobs.executor.shutdown(wait=True)
            client.__exit__(None, None, None)
        self.temp.cleanup()

    def install(self, *keys):
        for key in keys:
            directory = self.settings.multilingual_recognizer_dir(key)
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("model.bin", "config.json", "tokenizer.json"):
                (directory / name).touch()

    def client(self):
        app = create_app(self.settings, FakeProcessor())
        client = TestClient(app)
        client.__enter__()
        self.instances.append((app, client))
        return client

    def post(self, client, seconds=1.0, **data):
        return client.post("/api/jobs", data={"mode": "conversation", **data},
                           files={"file": ("talk.wav", wav_bytes(seconds), "audio/wav")})

    def test_health_is_ready_with_only_the_multilingual_models(self):
        self.install("turbo", "kazakh")
        health = self.client().get("/api/health").json()
        self.assertEqual(health["status"], "ready")
        self.assertTrue(health["features"]["conversation_mode"])
        self.assertFalse(health["features"]["two_speaker_separation"])
        self.assertEqual(health["limits"]["max_long_minutes"], 0.1)

    def test_models_missing_is_reported_before_upload_work(self):
        client = self.client()
        self.assertEqual(self.post(client).status_code, 503)
        self.install("turbo", "kazakh")
        self.assertEqual(self.post(client, model="cleartalk").status_code, 409)

    def test_a_fixed_language_needs_only_its_own_recognizer(self):
        self.install("turbo")  # no Kazakh model: Russian still works, automatic detection does not
        client = self.client()
        self.assertEqual(self.post(client, language="ru").status_code, 202)
        self.assertEqual(self.post(client, language="auto").status_code, 503)

    def test_inputs_are_validated(self):
        self.install("turbo", "kazakh")
        client = self.client()
        self.assertEqual(self.post(client, language="de").status_code, 422)
        self.assertEqual(self.post(client, speakers="9").status_code, 422)
        self.assertEqual(self.post(client, model="unknown").status_code, 422)

    def test_two_voices_needs_the_separator_and_conversation_mode(self):
        self.install("turbo", "kazakh")
        client = self.client()
        self.assertEqual(self.post(client, two_voices="true").status_code, 503)  # separator not installed
        self.settings.separator_dir.mkdir(parents=True)
        for name in ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"):
            (self.settings.separator_dir / name).touch()
        other_mode = client.post("/api/jobs", data={"mode": "transcribe", "two_voices": "true"},
                                 files={"file": ("talk.wav", wav_bytes(1.0), "audio/wav")})
        self.assertEqual(other_mode.status_code, 422)
        accepted = self.post(client, two_voices="true")
        self.assertEqual(accepted.status_code, 202, accepted.text)
        context = json.loads((self.settings.jobs_dir / accepted.json()["id"] / "input.json").read_text())
        self.assertTrue(context["two_voices"])

    def test_long_recordings_are_accepted_only_in_conversation_mode(self):
        self.install("turbo", "kazakh")
        client = self.client()
        accepted = self.post(client, seconds=4.0, language="auto", speakers="3")
        self.assertEqual(accepted.status_code, 202, accepted.text)
        self.assertEqual(self.post(client, seconds=7.0).status_code, 422)  # over the long limit too
        short_mode = client.post("/api/jobs", data={"mode": "transcribe"},
                                 files={"file": ("talk.wav", wav_bytes(4.0), "audio/wav")})
        self.assertIn(short_mode.status_code, {422, 503})
        job_id = accepted.json()["id"]
        context = json.loads((self.settings.jobs_dir / job_id / "input.json").read_text())
        self.assertEqual(context, {"is_demo": False, "model": "auto", "language": "auto", "speakers": 3})
        for _ in range(200):
            if client.get(f"/api/jobs/{job_id}").json()["status"] in {"done", "error"}:
                break
            time.sleep(0.01)


def test_pipeline_dispatches_conversation_jobs(tmp_path, monkeypatch):
    from backend.pipeline import AudioPipeline

    pipeline = AudioPipeline(Settings(root=tmp_path))
    seen = {}
    monkeypatch.setattr(pipeline.conversation, "process",
                        lambda job_id, job_dir, context, update: seen.update(context=context) or {"mode": "conversation"})
    (tmp_path / "input.json").write_text(json.dumps({"language": "kk", "model": "auto", "speakers": 2}))
    assert pipeline.process("a" * 32, tmp_path, "conversation", lambda stage, progress: None) == {"mode": "conversation"}
    assert seen["context"]["language"] == "kk" and pipeline.inference_verified
    assert SimpleNamespace  # keep the import used on every platform


def test_multilingual_demo_is_served_only_when_installed(tmp_path):
    settings = Settings(root=tmp_path)
    app = create_app(settings, FakeProcessor())
    with TestClient(app) as client:
        assert client.get("/api/demo/conversation").status_code == 404
        settings.demo_dir.mkdir(parents=True)
        (settings.demo_dir / "conversation.wav").write_bytes(wav_bytes())
        (settings.demo_dir / "conversation.json").write_text(json.dumps({"title": "Fixture demo"}))
        demo = client.get("/api/demo/conversation").json()
        assert demo["title"] == "Fixture demo" and demo["audio_url"] == "/media/demo/conversation.wav"
        assert client.get(demo["audio_url"]).status_code == 200
    app.state.jobs.executor.shutdown(wait=True)


def test_small_recognizer_stands_in_for_models_that_are_not_installed(tmp_path):
    settings = Settings(root=tmp_path)
    settings.multilingual_dir.mkdir(parents=True)
    (settings.multilingual_dir / "profile.json").write_text(json.dumps({**DEFAULT_MULTILINGUAL_PROFILE, "small_recognizer": "cleartalk"}))
    directory = settings.multilingual_recognizer_dir("cleartalk")
    directory.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (directory / name).touch()
    status = settings.multilingual_status()
    assert status["available"] and status["detector"] == "cleartalk"
    assert status["routes"] == {"kk": "cleartalk", "ru": "cleartalk", "en": "cleartalk"}
    assert status["preferred_routes"] == DEFAULT_MULTILINGUAL_PROFILE["routes"]
    pipeline = conversation.ConversationPipeline(settings)
    assert pipeline.route("auto", "kk") == "cleartalk" and pipeline.route("cleartalk", "ru") == "cleartalk"


def test_installed_routes_are_kept_and_a_partial_small_recognizer_is_refused(tmp_path):
    settings = Settings(root=tmp_path)
    for key in ("turbo", "kazakh"):
        directory = settings.multilingual_recognizer_dir(key)
        directory.mkdir(parents=True)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (directory / name).touch()
    assert settings.multilingual_status()["routes"] == {"kk": "kazakh", "ru": "turbo", "en": "turbo"}
    other = Settings(root=tmp_path / "other")
    other.multilingual_dir.mkdir(parents=True)
    (other.multilingual_dir / "profile.json").write_text(json.dumps({**DEFAULT_MULTILINGUAL_PROFILE, "small_recognizer": "kazakh"}))
    with pytest.raises(ValueError, match="Invalid multilingual"):
        other.multilingual_profile


def test_language_label_is_read_from_the_text_of_our_recognizer():
    from backend.multilingual import language_of_text

    assert language_of_text("Қазір болған аумаққа қызмет жұмылдырылды.", "ru") == "kk"
    assert language_of_text("Основной религией является православие.", "kk") == "ru"
    assert language_of_text("Some festivals have special camping areas.", "ru") == "en"
    assert language_of_text("Мен келдім", "kk") == "kk"          # short, no Kazakh-only letter: keep the detection
    assert language_of_text("включи Netflix на кухне", "ru") == "ru"
    assert language_of_text("1976", "en") == "en" and language_of_text("", "ru") == "ru"


def test_only_our_recognizer_relabels_turns(tmp_path, monkeypatch):
    found = lambda language: {"language": language, "uncertain": False}  # noqa: E731
    pipeline = make_pipeline(tmp_path, monkeypatch, [found("ru"), found("ru")])
    monkeypatch.setattr(conversation, "transcribe_window",
                        lambda model, audio, language: {"text": "Қазақша сөйлем осы жерде тұр.", "language": language,
                                                        "segments": [{"needs_review": False}], "speech_seconds": 1.0})
    ours = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="cleartalk")
    assert {turn["language"] for turn in ours["turns"]} == {"kk"}
    queue = [found("ru"), found("ru")]
    monkeypatch.setattr(conversation, "detect_language", lambda model, audio: queue.pop(0))
    other = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="small")
    assert {turn["language"] for turn in other["turns"]} == {"ru"}


# ----------------------------------------------------------------------------- languages that change inside a turn
def test_languages_of_a_turn_are_read_from_its_alphabets():
    from backend.multilingual import languages_of_text

    assert languages_of_text("Бүгін біз жаңа жобаны бастаймыз and the deadline is next Friday", "kk") == ["kk", "en"]
    assert languages_of_text("Сегодня мы обсудим the new release plan вместе", "ru") == ["ru", "en"]
    assert languages_of_text("Қазір болған аумаққа қызмет жұмылдырылды.", "ru") == ["kk"]
    assert languages_of_text("включи Netflix на кухне пожалуйста", "ru") == ["ru"]      # one borrowed word is not a second language
    assert languages_of_text("", "en") == ["en"]


def test_mixed_setting_sends_every_turn_to_the_mixed_recognizer(tmp_path, monkeypatch):
    found = lambda language: {"language": language, "uncertain": False}  # noqa: E731
    pipeline = make_pipeline(tmp_path, monkeypatch, [found("kk"), found("ru")])
    monkeypatch.setattr(conversation, "transcribe_window",
                        lambda model, audio, language: {"text": "Бүгін біз жобаны бастаймыз and the deadline is Friday", "language": language,
                                                        "segments": [{"needs_review": False}], "speech_seconds": 1.0})
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="mixed", model="auto")
    assert {turn["recognizer"] for turn in result["turns"]} == {"cleartalk"}
    assert all(turn["language"] == "mixed" and turn["languages"] == ["kk", "en"] for turn in result["turns"])
    assert result["turns"][0]["language_name"] == "Kazakh + English"
    queue = [found("kk"), found("ru")]
    monkeypatch.setattr(conversation, "detect_language", lambda model, audio: queue.pop(0))
    chosen = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="mixed", model="kazakh")
    assert {turn["recognizer"] for turn in chosen["turns"]} == {"cleartalk"}        # a Kazakh-only model cannot serve mixed turns


def test_mixed_setting_is_accepted_by_the_api_when_its_models_are_installed(tmp_path):
    settings = Settings(root=tmp_path, max_long_seconds=6)
    app = create_app(settings, FakeProcessor())
    with TestClient(app) as client:
        post = lambda: client.post("/api/jobs", data={"mode": "conversation", "language": "mixed"},  # noqa: E731
                                   files={"file": ("talk.wav", wav_bytes(1.0), "audio/wav")})
        assert post().status_code == 503
        for key in ("turbo", "cleartalk"):
            directory = settings.multilingual_recognizer_dir(key)
            directory.mkdir(parents=True)
            for name in ("model.bin", "config.json", "tokenizer.json"):
                (directory / name).touch()
        assert post().status_code == 202
        assert client.get("/api/health").json()["multilingual"]["mixed_recognizer"] == "cleartalk"
    app.state.jobs.executor.shutdown(wait=True)


# ----------------------------------------------------------------------------- check examples (corrected transcripts)
class ConversationFake(FakeProcessor):
    def process(self, job_id, job_dir, mode, update):
        result = super().process(job_id, job_dir, mode, update)
        result.update(mode="conversation", language="auto", model="auto", duration_seconds=1.0,
                      turns=[{"start": 0.0, "end": 1.0, "speaker": "Speaker 1", "language": "ru", "language_name": "Russian",
                              "text": "привет мир", "recognizer": "turbo", "needs_review": False}])
        return result


def checked_example_client(tmp_path, **overrides):
    settings = Settings(root=tmp_path, max_long_seconds=6, **overrides)
    for key in ("turbo", "kazakh"):
        directory = settings.multilingual_recognizer_dir(key)
        directory.mkdir(parents=True)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (directory / name).touch()
    app = create_app(settings, ConversationFake())
    return settings, app


def finished_job(client):
    job = client.post("/api/jobs", data={"mode": "conversation"}, files={"file": ("talk.wav", wav_bytes(1.0), "audio/wav")}).json()
    for _ in range(300):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.01)
    raise AssertionError("fixture job did not finish")


def test_a_corrected_transcript_is_kept_with_its_audio_only_with_consent(tmp_path):
    settings, app = checked_example_client(tmp_path)
    turns = [{"start": 0.0, "end": 1.0, "speaker": "Алия", "language": "ru", "text": "Привет, мир!"}]
    with TestClient(app) as client:
        job = finished_job(client)
        assert job["status"] == "done"
        url = f"/api/jobs/{job['id']}/example"
        assert client.post(url, json={"turns": turns}).status_code == 422                       # no consent
        assert client.post(url, json={"turns": [{**turns[0], "language": "de"}], "consent": True}).status_code == 422
        assert client.post(url, json={"turns": [{**turns[0], "end": -1}], "consent": True}).status_code == 422
        saved = client.post(url, json={"turns": turns, "consent": True, "note": "кухня, два человека"})
        assert saved.status_code == 200 and saved.json() == {"saved": True, "examples": 1}
        folder = settings.examples_dir / job["id"]
        reference = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
        assert reference["turns"] == turns and reference["first_transcript"]["turns"][0]["text"] == "привет мир"
        assert (folder / "audio.wav").stat().st_size > 1000
        other = TestClient(app)                                                                  # another browser
        assert other.post(url, json={"turns": turns, "consent": True}).status_code == 404
    app.state.jobs.executor.shutdown(wait=True)


def test_check_examples_are_refused_on_a_shared_deployment(tmp_path):
    settings, app = checked_example_client(tmp_path, deployment="server")
    with TestClient(app) as client:
        job = finished_job(client)
        turns = [{"start": 0.0, "end": 1.0, "speaker": "Speaker 1", "language": "ru", "text": "Привет"}]
        assert client.post(f"/api/jobs/{job['id']}/example", json={"turns": turns, "consent": True}).status_code == 403
    assert not settings.examples_dir.exists()
    app.state.jobs.executor.shutdown(wait=True)


# ----------------------------------------------------------------------------- graphics memory
def test_turns_are_transcribed_recognizer_by_recognizer_and_returned_in_spoken_order(tmp_path, monkeypatch):
    found = lambda language: {"language": language, "uncertain": False}  # noqa: E731
    pipeline = make_pipeline(tmp_path, monkeypatch, [found("kk"), found("ru")])
    calls, stages = [], []
    monkeypatch.setattr(conversation, "transcribe_window",
                        lambda model, audio, language: calls.append(model.name) or {"text": f"{model.name}:{language}", "language": language,
                                                                                   "segments": [{"needs_review": False}], "speech_seconds": 1.0})
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto",
                                 update=lambda stage, progress: stages.append(stage))
    assert calls == ["kazakh", "kazakh", "turbo"]                    # one recognizer at a time
    assert [t["text"] for t in result["turns"]] == ["kazakh:kk", "turbo:ru", "kazakh:kk"]   # spoken order
    assert [t["start"] for t in result["turns"]] == [0.0, 4.5, 9.2] and "order" not in result["turns"][0]
    assert [stage for stage in stages if stage.startswith("Transcribing")] == [f"Transcribing turn {n} of 3" for n in (1, 2, 3)]


def test_recognizers_that_do_not_fit_in_graphics_memory_are_unloaded_oldest_first(tmp_path, monkeypatch):
    import faster_whisper

    settings = Settings(root=tmp_path)
    sizes = {"turbo": 1600, "kazakh": 1600, "cleartalk-large": 1600, "cleartalk": 480}
    for key, size in sizes.items():
        directory = settings.multilingual_recognizer_dir(key)
        directory.mkdir(parents=True)
        (directory / "model.bin").write_bytes(b"0" * size)
    events = []

    class Loaded:
        def __init__(self, path, **options):
            self.name = Path(path).name
            self.model = self
            events.append(("load", self.name))

        def unload_model(self):
            events.append(("unload", self.name))

    monkeypatch.setattr(faster_whisper, "WhisperModel", Loaded)
    monkeypatch.setattr(Settings, "verify_multilingual_recognizer", lambda self, key: None)
    pipeline = conversation.ConversationPipeline(settings)
    monkeypatch.setattr(pipeline, "device", lambda: ("cuda", "int8_float32"))
    pipeline._memory_limit = 5000          # as on a 6 GB card: two large recognizers and a small one, not three large
    for key in ("turbo", "kazakh", "cleartalk"):
        pipeline.recognizer(key)
    assert pipeline.loaded_models()["recognizers"] == ["cleartalk", "kazakh", "turbo"] and not [e for e in events if e[0] == "unload"]
    pipeline.recognizer("turbo")            # used again: now the Kazakh model is the one used longest ago
    pipeline.recognizer("cleartalk-large")
    assert [e for e in events if e[0] == "unload"] == [("unload", "kazakh-whisper-large-v3-turbo")]
    assert pipeline.loaded_models()["recognizers"] == ["cleartalk", "cleartalk-large", "turbo"]
    pipeline.recognizer("kazakh")           # the small model goes first, then the large one used longest ago
    assert [name for kind, name in events if kind == "unload"][1:] == ["whisper-small-cleartalk-multilingual", "whisper-large-v3-turbo"]
    pipeline._memory_limit = None           # a CPU has no such limit: everything stays loaded
    for key in sizes:
        pipeline.recognizer(key)
    assert len(pipeline.loaded_models()["recognizers"]) == 4


# ----------------------------------------------------------------------------- decoding options of the profile
def test_an_uncertain_turn_goes_to_the_recognizer_of_the_mixed_setting_when_the_option_is_on(tmp_path, monkeypatch):
    found = lambda language, uncertain=False: {"language": language, "uncertain": uncertain}  # noqa: E731
    pipeline = make_pipeline(tmp_path, monkeypatch, [found("en", uncertain=True), found("ru")])
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    audio = np.zeros(16000 * 10, dtype=np.float32)
    plain = pipeline.transcribe(audio, language="auto", model="auto", uncertain_to_mixed=False)
    assert [t["recognizer"] for t in plain["turns"]] == ["turbo", "turbo", "turbo"]
    pipeline = make_pipeline(tmp_path, monkeypatch, [found("en", uncertain=True), found("ru")])
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    safe = pipeline.transcribe(audio, language="auto", model="auto", uncertain_to_mixed=True)
    # the uncertain turn and the short one (no detection of its own) go to the recognizer that writes what it hears
    assert [t["recognizer"] for t in safe["turns"]] == ["cleartalk-large", "turbo", "cleartalk-large"]


def test_the_mixed_setting_can_skip_the_language_detector(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [])
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    monkeypatch.setattr(conversation, "detect_language", lambda model, audio: (_ for _ in ()).throw(AssertionError("detector ran")))
    monkeypatch.setattr(conversation, "transcribe_window",
                        lambda model, audio, language: {"text": "Сәлем достар" if language is None else "?", "language": "kk",
                                                        "segments": [{"needs_review": False}], "speech_seconds": 1.0})
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="mixed", model="auto", skip_detection=True)
    assert [t["language"] for t in result["turns"]] == ["kk", "kk", "kk"]          # the label is read from the text
    assert {t["recognizer"] for t in result["turns"]} == {"cleartalk-large"}


def test_the_options_come_from_the_profile(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path)
    profile = json.loads(json.dumps(conversation.Settings(root=tmp_path).multilingual_profile))
    profile["options"] = {"uncertain_to_mixed": True}
    settings.multilingual_dir.mkdir(parents=True, exist_ok=True)
    (settings.multilingual_dir / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    seen = {}
    pipeline = conversation.ConversationPipeline(Settings(root=tmp_path))
    monkeypatch.setattr(pipeline, "recognizer", lambda key: FakeRecognizer(key))
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: seen.setdefault("mixed", "cleartalk-large"))
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [speakers.Turn(0.0, 4.0, 0)])
    monkeypatch.setattr(conversation, "detect_language", lambda model, audio: {"language": "en", "uncertain": True})
    monkeypatch.setattr(conversation, "transcribe_window", lambda model, audio, language: {"text": f"{model.name}", "language": language,
                                                                                           "segments": [{"needs_review": False}], "speech_seconds": 1.0})
    result = pipeline.transcribe(np.zeros(16000 * 5, dtype=np.float32), language="auto", model="auto")
    assert result["turns"][0]["recognizer"] == "cleartalk-large"


def test_an_english_turn_that_our_model_hears_as_kazakh_keeps_our_text_when_cross_checked(tmp_path, monkeypatch):
    found = lambda language: {"language": language, "uncertain": False}  # noqa: E731
    pipeline = make_pipeline(tmp_path, monkeypatch, [found("en"), found("en")])
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    # Turns are written in spoken order by large-v3-turbo, each followed by its cross-check: the n-th call of a model is turn n.
    texts = {"turbo": ["we are going to talk about it", "One of the strong points", "Thank you"],
             "cleartalk-large": ["Біздің сөзіміз ешкімге өтпейді", "One of the strong points", "Thank you"]}
    calls = {"turbo": 0, "cleartalk-large": 0}

    def window(model, audio, language):
        text = texts[model.name][calls[model.name]]
        calls[model.name] += 1
        return {"text": text, "language": language, "segments": [{"needs_review": False}], "speech_seconds": 1.0}

    monkeypatch.setattr(conversation, "transcribe_window", window)
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto", cross_check=True,
                                 uncertain_to_mixed=False)
    first = result["turns"][0]
    assert first["recognizer"] == "cleartalk-large" and first["language"] == "kk" and first["text"].startswith("Біздің")
    assert any(t["recognizer"] == "turbo" for t in result["turns"][1:])        # real English stays with large-v3-turbo


def test_the_kazakh_cross_check_takes_over_only_turns_written_in_kazakh(tmp_path, monkeypatch):
    found = lambda language: {"language": language, "uncertain": False}  # noqa: E731
    pipeline = make_pipeline(tmp_path, monkeypatch, [found("en"), found("en")])
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    texts = {"turbo": ["we are going to talk about it", "simply the sound", "Thank you"],
             "cleartalk-large": ["Біздің сөзіміз ешкімге өтпейді", "Симпли зе саунд ми ок", "Thank you"]}
    calls = {"turbo": 0, "cleartalk-large": 0}

    def window(model, audio, language):
        text = texts[model.name][calls[model.name]]
        calls[model.name] += 1
        return {"text": text, "language": language, "segments": [{"needs_review": False}], "speech_seconds": 1.0}

    monkeypatch.setattr(conversation, "transcribe_window", window)
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto", cross_check="kazakh",
                                 uncertain_to_mixed=False)
    assert result["turns"][0]["recognizer"] == "cleartalk-large"                 # Kazakh letters: taken over
    assert result["turns"][1]["recognizer"] == "turbo"                           # English written in Russian letters: kept


def test_by_region_recognises_a_whole_stretch_and_gives_each_sentence_to_its_speaker(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "ru", "uncertain": False}])
    # Speaker labels flip inside one stretch of speech (0-9 s); the recognizer must still get the stretch whole.
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [speakers.Turn(0.0, 4.0, 0), speakers.Turn(4.0, 6.0, 1),
                                                                                     speakers.Turn(6.0, 9.0, 0)])
    monkeypatch.setattr(conversation, "speech_regions", lambda audio: [(0.0, 9.0)])
    pieces = []

    def window(model, audio, language):
        pieces.append(len(audio) / 16000)
        return {"text": "Первое предложение. Второе. Третье.", "language": language, "speech_seconds": 9.0,
                "segments": [{"start": 0.2, "end": 3.9, "text": "Первое предложение.", "needs_review": False},
                             {"start": 4.2, "end": 5.9, "text": "Второе.", "needs_review": False},
                             {"start": 6.1, "end": 9.0, "text": "Третье.", "needs_review": True}]}

    monkeypatch.setattr(conversation, "transcribe_window", window)
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="ru", model="auto", by_region=True)
    assert len(pieces) == 1 and pieces[0] > 9                                   # one call for the whole stretch
    assert [(t["speaker"], t["text"]) for t in result["turns"]] == [("Speaker 1", "Первое предложение."), ("Speaker 2", "Второе."),
                                                                     ("Speaker 1", "Третье.")]
    assert [t["needs_review"] for t in result["turns"]] == [False, False, True]


def _voices(monkeypatch, pipeline, mapping=None):
    """The speaker check sees two different people (or ``mapping`` when given) without loading the voice model."""
    monkeypatch.setattr(pipeline, "encoder", lambda: None)
    monkeypatch.setattr(conversation, "voice_print", lambda *args, **kwargs: np.ones(2) / np.sqrt(2))
    monkeypatch.setattr(conversation, "who_is_who", lambda tracks, people: {0: 0, 1: 1} if mapping is None else mapping)


def test_two_people_talking_at_once_are_separated_and_each_voice_goes_to_its_speaker(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "kk", "uncertain": False}, {"language": "ru", "uncertain": False}])
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [speakers.Turn(0.0, 5.0, 0), speakers.Turn(5.0, 9.0, 1)])
    monkeypatch.setattr(conversation, "speech_regions", lambda audio: [(0.0, 9.0)])
    monkeypatch.setattr(type(pipeline.settings), "separator_dir", property(lambda self: tmp_path))

    class FakeSeparator:
        def split(self, audio):
            quiet, loud = np.zeros(len(audio), np.float32), np.ones(len(audio), np.float32)
            first = np.concatenate([loud[:16000 * 5], quiet[16000 * 5:]])     # track 0 carries the first speaker's solo part
            return [first, 1 - first], [first, 1 - first], True, {}

        def dominant(self, tracks, spans):
            return 0

        own_voice = staticmethod(Separator.own_voice)

    monkeypatch.setattr(pipeline, "separator", lambda: FakeSeparator())
    _voices(monkeypatch, pipeline)
    calls = []

    def window(model, audio, language):
        calls.append(model.name)
        text = "Мен қазақша сөйлеймін" if len(calls) == 1 else "А я говорю по-русски"
        return {"text": text, "language": language, "speech_seconds": 4.0, "segments": [{"start": 0.5, "end": 8.0, "text": text, "needs_review": False}]}

    monkeypatch.setattr(conversation, "transcribe_window", window)
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto", separate_overlaps=True)
    assert calls == ["cleartalk-large", "cleartalk-large"]                     # one recognition per separated voice
    assert [(t["speaker"], t["language"]) for t in result["turns"]] == [("Speaker 1", "kk"), ("Speaker 2", "ru")]
    assert all(t["needs_review"] for t in result["turns"])                     # separated speech is flagged for checking


def test_a_separated_track_that_only_carries_leakage_writes_nothing(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "kk", "uncertain": False}, {"language": "ru", "uncertain": False}])
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [speakers.Turn(0.0, 5.0, 0), speakers.Turn(5.0, 9.0, 1)])
    monkeypatch.setattr(conversation, "speech_regions", lambda audio: [(0.0, 9.0)])
    monkeypatch.setattr(type(pipeline.settings), "separator_dir", property(lambda self: tmp_path))
    voice = np.ones(16000 * 10, np.float32)

    class LeakySeparator:
        def split(self, audio):
            return [voice, voice * 0.05], [voice, voice], True, {}             # track 1 is 26 dB quieter everywhere

        def dominant(self, tracks, spans):
            return 0

        own_voice = staticmethod(Separator.own_voice)

    monkeypatch.setattr(pipeline, "separator", lambda: LeakySeparator())
    _voices(monkeypatch, pipeline)

    words = iter(["бірінші сөз", "басқа мәтін"])

    def window(model, audio, language):
        text = next(words)
        return {"text": text, "language": language, "speech_seconds": 4.0, "segments": [{"start": 0.5, "end": 8.0, "text": text, "needs_review": False}]}

    monkeypatch.setattr(conversation, "transcribe_window", window)
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto", separate_overlaps=True)
    assert [t["speaker"] for t in result["turns"]] == ["Speaker 1"]          # the leaked copy is not written for Speaker 2


def test_when_separation_yields_nothing_the_stretch_is_recognised_as_usual(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "kk", "uncertain": False}, {"language": "ru", "uncertain": False}])
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [speakers.Turn(0.0, 5.0, 0), speakers.Turn(5.0, 9.0, 1)])
    monkeypatch.setattr(conversation, "speech_regions", lambda audio: [(0.0, 9.0)])
    monkeypatch.setattr(type(pipeline.settings), "separator_dir", property(lambda self: tmp_path))
    silence = np.zeros(16000 * 10, np.float32)

    class EmptySeparator:
        def split(self, audio):
            return [silence, silence], [silence, silence], True, {}

        def dominant(self, tracks, spans):
            return None

        own_voice = staticmethod(Separator.own_voice)

    monkeypatch.setattr(pipeline, "separator", lambda: EmptySeparator())
    _voices(monkeypatch, pipeline)
    calls = []

    def window(model, audio, language):
        calls.append(model.name)
        return {"text": "Сәлем", "language": language, "speech_seconds": 4.0, "segments": [{"start": 0.5, "end": 4.0, "text": "Сәлем", "needs_review": False}]}

    monkeypatch.setattr(conversation, "transcribe_window", window)
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto", separate_overlaps=True)
    assert len(calls) == 4                                                      # two empty tracks, then the two turns as usual
    assert [t["speaker"] for t in result["turns"]] == ["Speaker 1", "Speaker 2"]


def _three_turn_recording(floor):
    rng = np.random.default_rng(0)
    audio = floor * rng.standard_normal(16000 * 10).astype(np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(16000 * 3) / 16000).astype(np.float32)
    audio[16000 * 1:16000 * 4] += tone
    audio[16000 * 5:16000 * 6] += 0.03 * tone[:16000]                         # a short stretch 20 dB quieter
    audio[16000 * 7:16000 * 10] += tone
    return audio


@pytest.mark.parametrize("floor, hidden", [(0.01, 1), (0.0, 0)])
def test_short_quiet_stretches_near_the_background_level_are_set_aside(tmp_path, monkeypatch, floor, hidden):
    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "ru", "uncertain": False}] * 3)
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [
        speakers.Turn(1.0, 4.0, 0), speakers.Turn(5.0, 6.0, 1), speakers.Turn(7.0, 10.0, 0)])
    words = iter(["Сәлем", "Эпизод первого сезона", "Қалайсыз"])

    def window(model, audio, language):
        text = next(words)
        return {"text": text, "language": language, "speech_seconds": 1.0, "segments": [{"start": 0.1, "end": 0.9, "text": text, "needs_review": False}]}

    monkeypatch.setattr(conversation, "transcribe_window", window)
    result = pipeline.transcribe(_three_turn_recording(floor), language="auto", model="auto", hide_background=True)
    assert len(result["background_turns"]) == hidden            # in noise the quiet stretch is background; in silence it is kept
    assert len(result["turns"]) == 3 - hidden
    if hidden:
        assert result["background_turns"][0]["text"] == "Эпизод первого сезона"


def test_region_first_closes_breath_pauses_but_keeps_real_gaps():
    regions = [(0.0, 1.0), (1.3, 2.0), (3.0, 4.0), (4.5, 30.0), (30.2, 31.0)]
    assert speakers.joined_regions(regions) == [(0.0, 2.0), (3.0, 4.0), (4.5, 30.0), (30.2, 31.0)]   # never past 24 s by joining


def test_only_the_stretch_where_two_people_talk_is_separated(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "ru", "uncertain": False}] * 3)
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [
        speakers.Turn(0.0, 4.0, 0), speakers.Turn(5.0, 7.0, 1), speakers.Turn(7.0, 9.0, 0)])
    monkeypatch.setattr(conversation, "speech_regions", lambda audio: [(0.0, 4.0), (5.0, 9.0)])
    monkeypatch.setattr(type(pipeline.settings), "separator_dir", property(lambda self: tmp_path))
    split_lengths = []

    class HalfSeparator:
        def split(self, audio):
            split_lengths.append(len(audio) / 16000)
            first = np.zeros(len(audio), np.float32); first[: len(audio) // 2] = 1.0
            return [first, 1 - first], [first, 1 - first], True, {}

        def dominant(self, tracks, spans):
            return 0

        own_voice = staticmethod(Separator.own_voice)

    monkeypatch.setattr(pipeline, "separator", lambda: HalfSeparator())
    _voices(monkeypatch, pipeline)
    texts = iter(["первая", "вторая", "одна"])

    def window(model, audio, language):
        text = next(texts)
        start = 3.0 if text == "вторая" else 0.2                                 # each voice speaks in its own half
        return {"text": text, "language": language, "speech_seconds": 1.0, "segments": [{"start": start, "end": start + 0.8, "text": text, "needs_review": False}]}

    monkeypatch.setattr(conversation, "transcribe_window", window)
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto", separate_overlaps=True)
    assert len(split_lengths) == 1 and 4.0 <= split_lengths[0] <= 4.5          # only 5-9 s went through the separator
    assert [(t["text"], bool(t.get("overlap"))) for t in result["turns"]] == [("одна", False), ("первая", True), ("вторая", True)]


def test_region_first_does_not_join_two_people_across_a_pause():
    regions = [(0.0, 3.0), (3.3, 6.0), (6.3, 9.0)]
    speaker = {0.0: 0, 3.3: 0, 6.3: 1}
    joined = speakers.joined_regions(regions, same=lambda a, b: speaker[b[0]] == speaker[a[0]] if a[0] in speaker else True)
    assert joined == [(0.0, 6.0), (6.3, 9.0)]


def test_one_voice_written_twice_is_not_taken_for_two_people(tmp_path, monkeypatch):
    from backend.overlap import same_words

    assert same_words("Включи мне на смотрешке передачу рэн тв.", "включи мне на смотрешке передачу")
    assert not same_words("Мен қазақша сөйлеймін", "А я говорю по-русски")
    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "ru", "uncertain": False}] * 2)
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [speakers.Turn(0.0, 5.0, 0), speakers.Turn(5.0, 9.0, 1)])
    monkeypatch.setattr(conversation, "speech_regions", lambda audio: [(0.0, 9.0)])
    monkeypatch.setattr(type(pipeline.settings), "separator_dir", property(lambda self: tmp_path))
    voice = np.ones(16000 * 10, np.float32)

    class CopyingSeparator:
        def split(self, audio):
            return [voice, voice], [voice, voice], True, {}                     # the same voice on both tracks

        def dominant(self, tracks, spans):
            return 0

        own_voice = staticmethod(Separator.own_voice)

    monkeypatch.setattr(pipeline, "separator", lambda: CopyingSeparator())
    _voices(monkeypatch, pipeline)
    monkeypatch.setattr(conversation, "transcribe_window", lambda model, audio, language: {
        "text": "Включи мне передачу", "language": language, "speech_seconds": 3.0,
        "segments": [{"start": 0.5, "end": 3.0, "text": "Включи мне передачу", "needs_review": False}]})
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto", separate_overlaps=True)
    assert not any(t.get("overlap") for t in result["turns"])                  # recognised as usual, not twice


def test_speakers_are_numbered_in_the_order_they_first_speak(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "ru", "uncertain": False}] * 3)
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [
        speakers.Turn(0.0, 3.0, 4), speakers.Turn(3.5, 6.0, 1), speakers.Turn(6.5, 9.0, 4)])
    monkeypatch.setattr(conversation, "transcribe_window", lambda model, audio, language: {
        "text": "сөз", "language": language, "speech_seconds": 2.0, "segments": [{"start": 0.1, "end": 2.0, "text": "сөз", "needs_review": False}]})
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto")
    assert [t["speaker"] for t in result["turns"]] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_text_first_lets_our_model_write_and_sends_only_english_turns_to_the_english_route(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [])
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    monkeypatch.setattr(pipeline, "route", lambda model, language: "turbo" if language == "en" else "cleartalk-large")
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [speakers.Turn(0.0, 3.0, 0), speakers.Turn(3.5, 7.0, 1)])
    monkeypatch.setattr(conversation, "detect_language", lambda model, audio: (_ for _ in ()).throw(AssertionError("detector must not run")))
    calls = []

    def window(model, audio, language):
        calls.append((model.name, language))
        if model.name == "turbo":
            text = "Hello there, how are you?"
        else:
            text = "Сәлеметсіз бе, қалайсыз?" if len(calls) == 1 else "hello there how are you"
        return {"text": text, "language": language, "speech_seconds": 2.0, "segments": [{"start": 0.1, "end": 2.0, "text": text, "needs_review": False}]}

    monkeypatch.setattr(conversation, "transcribe_window", window)
    result = pipeline.transcribe(np.zeros(16000 * 8, dtype=np.float32), language="auto", model="auto", text_first=True)
    assert calls == [("cleartalk-large", None), ("cleartalk-large", None), ("turbo", "en")]
    assert [(t["language"], t["recognizer"]) for t in result["turns"]] == [("kk", "cleartalk-large"), ("en", "turbo")]
    assert result["turns"][1]["text"] == "Hello there, how are you?"
    assert not any(t["needs_review"] for t in result["turns"])



def test_the_speaker_check_wants_two_different_people_of_the_conversation(monkeypatch):
    from backend import overlap

    monkeypatch.setattr(overlap, "MIN_VOICE_MATCH", 0.5)
    monkeypatch.setattr(overlap, "MIN_VOICE_MARGIN", 0.1)
    a, b, babble = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    assert overlap.who_is_who([b, a], [a, b]) == {0: 1, 1: 0}                 # two people: each track to its own speaker
    assert overlap.who_is_who([a, a * 0.9 + b * 0.1], [a, b]) is None          # one voice spread over both tracks
    assert overlap.who_is_who([a, babble], [a, b]) is None                     # background talkers as the "second voice"
    assert overlap.who_is_who([a, None], [a, b]) is None


def test_separation_is_skipped_when_the_tracks_are_not_two_people(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "ru", "uncertain": False}] * 2)
    monkeypatch.setattr(pipeline, "mixed_route", lambda model: "cleartalk-large")
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [speakers.Turn(0.0, 5.0, 0), speakers.Turn(5.0, 9.0, 1)])
    monkeypatch.setattr(conversation, "speech_regions", lambda audio: [(0.0, 9.0)])
    monkeypatch.setattr(type(pipeline.settings), "separator_dir", property(lambda self: tmp_path))
    voice = np.ones(16000 * 10, np.float32)

    class AnySeparator:
        def split(self, audio):
            return [voice, voice * 0.5], [voice, voice], True, {}

    monkeypatch.setattr(pipeline, "separator", lambda: AnySeparator())
    _voices(monkeypatch, pipeline, mapping=None)
    monkeypatch.setattr(conversation, "who_is_who", lambda tracks, people: None)
    calls = []
    monkeypatch.setattr(conversation, "transcribe_window", lambda model, audio, language: calls.append(1) or {
        "text": "сөз", "language": language, "speech_seconds": 2.0, "segments": [{"start": 0.1, "end": 2.0, "text": "сөз", "needs_review": False}]})
    result = pipeline.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="auto", model="auto", separate_overlaps=True)
    assert len(calls) == 2 and not any(t.get("overlap") for t in result["turns"])   # no track was transcribed
