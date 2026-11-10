"""ClearTalk inference API with private, persistent recording sessions."""

import asyncio
import contextlib
import hmac
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .audio import AudioValidationError, decode_limited
from .config import LANGUAGES, RECOGNIZERS, Settings
from .jobs import JOB_ID, JobManager, transcript_text
from .overlap import MAX_SECONDS as OVERLAP_MAX_SECONDS
from .pipeline import AudioPipeline
from .security import ServiceMiddleware, SessionSecurity, error_response

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm"}
MEDIA_FILES = {"original.wav", "source-a.wav", "source-b.wav"}
ERROR_CODES = {400: "bad_request", 401: "access_required", 403: "forbidden", 404: "not_found", 409: "not_ready", 413: "upload_too_large", 422: "invalid_input", 429: "queue_full", 503: "models_unavailable", 507: "storage_full"}
SERVICE_VERSION = "1.0.1"


class FreshStaticFiles(StaticFiles):
    """Static files the browser must revalidate (cheap: unchanged files answer 304)."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class AccessRequest(BaseModel):
    token: str = Field(max_length=4096)


class CheckedTurn(BaseModel):
    start: float = Field(ge=0, le=4 * 3600)
    end: float = Field(ge=0, le=4 * 3600)
    speaker: str = Field(min_length=1, max_length=80)
    language: str = Field(pattern="^(kk|ru|en|mixed)$")
    text: str = Field(max_length=4000)


class CheckedExample(BaseModel):
    """A transcript corrected by a person, kept with its audio as a check example."""
    turns: list[CheckedTurn] = Field(min_length=1, max_length=5000)
    consent: bool = False
    note: str = Field(default="", max_length=500)


def create_app(settings: Settings | None = None, processor=None) -> FastAPI:
    settings = settings or Settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    security = SessionSecurity(settings)
    pipeline = processor or AudioPipeline(settings)
    manager = JobManager(settings, pipeline)

    @asynccontextmanager
    async def lifespan(_app):
        manager.start()

        async def reap():
            while True:
                await asyncio.sleep(min(60, settings.job_ttl_seconds))
                await asyncio.to_thread(manager.cleanup)

        task = asyncio.create_task(reap())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            manager.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="ClearTalk", version=SERVICE_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.jobs = manager
    app.state.security = security
    app.add_middleware(ServiceMiddleware, settings=settings, security=security, manager=manager)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request, exc):
        message = exc.detail if isinstance(exc.detail, str) else "The request could not be processed."
        return error_response(exc.status_code, ERROR_CODES.get(exc.status_code, "request_failed"), message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _exc):
        return error_response(422, "invalid_input", "The request contains invalid or missing fields.")

    @app.exception_handler(Exception)
    async def unexpected_error(_request, exc):
        logging.getLogger(__name__).error("Unhandled service request error", exc_info=exc)
        return error_response(500, "internal_error", "The service could not complete this request. Please try again.")

    def model_status():
        models = settings.model_status()
        loaded = pipeline.loaded_models() if hasattr(pipeline, "loaded_models") else {}
        for name in models:
            models[name]["loaded"] = loaded.get(name, False)
        return models

    @app.get("/api/session")
    def session(request: Request):
        return {"access_required": bool(settings.access_token), "authenticated": request.state.authenticated, "retention_hours": settings.job_ttl_seconds / 3600}

    @app.post("/api/access")
    def grant_access(payload: AccessRequest, request: Request):
        if settings.access_token and not hmac.compare_digest(settings.access_token.encode(), payload.token.encode()):
            raise HTTPException(401, "The access token is incorrect.")
        response = JSONResponse({"authenticated": True})
        response.set_cookie(security.access_cookie, security.access_value(request.state.owner), **security.cookie_options())
        return response

    def conversation_needs(model: str, language: str) -> set[str]:
        """Recognizers a conversation job loads: the chosen one, routes for other languages, the detector."""
        profile = settings.multilingual_profile
        status = settings.multilingual_status()  # routes and detector among the installed recognizers
        if language == "mixed":  # the detector picks a language token; one recognizer writes every turn
            covers_all = model != "auto" and set(profile["recognizers"][model]["languages"]) == set(LANGUAGES)
            return {status["detector"], model if covers_all else status["mixed_recognizer"]}
        languages = LANGUAGES if language == "auto" else (language,)
        needed = {status["detector"]} if language == "auto" else set()
        for item in languages:
            covers = model != "auto" and item in profile["recognizers"][model]["languages"]
            needed.add(model if covers else status["routes"][item])
        return needed

    @app.get("/api/health")
    def health():
        models = model_status()
        default = settings.default_model
        multilingual = settings.multilingual_status()
        english = models[default]["available"] and models["separator"]["available"]
        installed = english or multilingual["available"]
        loaded = models[default]["loaded"] and models["separator"]["loaded"]
        return {
            "app": "ClearTalk",
            "status": "ready" if installed else "setup_required",
            "readiness": "loaded" if loaded else ("not_loaded" if installed else "setup_required"),
            "inference_verified": bool(getattr(pipeline, "inference_verified", False)),
            "version": SERVICE_VERSION,
            "model_profile_version": settings.active_profile["version"],
            "default_model": default,
            "models": models,
            "multilingual": multilingual,
            "limits": {"max_duration_seconds": settings.max_duration_seconds, "max_upload_mb": settings.max_upload_bytes // (1024 * 1024), "retention_hours": settings.job_ttl_seconds / 3600, "max_pending_jobs": settings.max_pending_jobs,
                       "max_long_minutes": settings.max_long_seconds / 60, "max_long_upload_mb": settings.max_long_upload_bytes // (1024 * 1024)},
            "busy": manager.pending() > 0,
            "pending_jobs": manager.pending(),
            "local_processing": settings.deployment == "local",
            "deployment": settings.deployment,
            "language": "English",
            "languages": {"two_voice_modes": ["English"], "conversation": ["Kazakh", "Russian", "English"]},
            "features": {"audio_upload": True, "two_speaker_separation": english, "transcription": True, "cleartalk_model": models["cleartalk"]["available"],
                         "conversation_mode": multilingual["available"], "speaker_labels": multilingual["speaker_labels"],
                         "visual_speech": False, "speaker_identification": False},
        }

    @app.get("/api/models")
    def available_models():
        return {"default": settings.default_model, "models": model_status()}

    @app.get("/api/demo")
    def demo():
        if not (settings.demo_dir / "mix.wav").is_file():
            raise HTTPException(404, "The demo recording is not installed yet. Run the setup script.")
        metadata_path = settings.demo_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        return {"title": metadata.get("title", "Two voices in noise"), "description": metadata.get("description", "A short, reproducible two-speaker demonstration."), "audio_url": "/media/demo/mix.wav", "reference_sources": metadata.get("reference_sources", []), "metadata": metadata}

    @app.get("/api/demo/conversation")
    def conversation_demo():
        """The optional multilingual demo: six readers, three languages (scripts/prepare_multilingual_demo.py)."""
        if not (settings.demo_dir / "conversation.wav").is_file():
            raise HTTPException(404, "The multilingual demo recording is not installed.")
        metadata_path = settings.demo_dir / "conversation.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        return {"title": metadata.get("title", "Three languages, six readers"), "description": metadata.get("description", ""),
                "audio_url": "/media/demo/conversation.wav", "metadata": metadata}

    @app.get("/api/demo/overlap")
    def overlap_demo():
        path = settings.root / "samples/two_voices_kk_ru_en/kazakh_and_russian.wav"
        if not path.is_file():
            raise HTTPException(404, "The demo recording is not installed yet.")
        return {"title": "Kazakh + Russian: overlapping voices", "audio_url": "/media/overlap-demo.wav",
                "description": "Controlled demonstration, not an accuracy benchmark.", "duration_seconds": 9.72}

    @app.get("/media/overlap-demo.wav")
    def overlap_demo_audio():
        path = settings.root / "samples/two_voices_kk_ru_en/kazakh_and_russian.wav"
        if not path.is_file():
            raise HTTPException(404, "Audio file not found.")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/media/demo/{filename}")
    def demo_media(filename: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.wav", filename):
            raise HTTPException(404, "Audio file not found.")
        target = settings.demo_dir / filename
        if not target.is_file():
            raise HTTPException(404, "Audio file not found.")
        return FileResponse(target, media_type="audio/wav")

    @app.post("/api/jobs", status_code=202)
    async def submit_job(request: Request, file: UploadFile | None = File(None), demo: bool = Form(False), mode: str = Form("separate"), model: str | None = Form(None),
                         language: str = Form("auto"), speakers: int = Form(0), two_voices: bool = Form(False)):
        if mode not in {"separate", "transcribe", "conversation"}:
            raise HTTPException(422, "Mode must be separate, transcribe or conversation.")
        conversation = mode == "conversation"
        if demo and file is not None:
            raise HTTPException(422, "Choose either the demo or an uploaded file.")
        if not demo and file is None:
            raise HTTPException(422, "Choose an audio file or use the demo.")
        if two_voices and not conversation:
            raise HTTPException(422, "Two people at the same time is a setting of Conversation mode.")
        if conversation:
            multilingual = settings.multilingual_status()
            model = model or "auto"
            if model != "auto" and model not in multilingual["recognizers"]:
                raise HTTPException(422, "Model must be auto or one of the multilingual recognizers.")
            if language not in ("auto", "mixed") and language not in LANGUAGES:
                raise HTTPException(422, "Language must be auto, mixed, kk, ru or en.")
            if not 0 <= speakers <= 8:
                raise HTTPException(422, "The number of speakers must be between 1 and 8, or 0 to find it automatically.")
            missing = sorted(key for key in conversation_needs(model, language) if not multilingual["recognizers"][key]["available"])
            if missing:
                if model != "auto" and model in missing:
                    raise HTTPException(409, f"{multilingual['recognizers'][model]['name']} is not installed. Select another recognizer.")
                raise HTTPException(503, "The multilingual speech models are not installed yet. Complete setup and restart ClearTalk.")
            upload_limit, duration_limit = settings.max_long_upload_bytes, settings.max_long_seconds
            if two_voices:
                if not settings.model_status()["separator"]["available"]:
                    raise HTTPException(503, "The separation model is not installed yet. Complete setup and restart ClearTalk.")
                duration_limit = min(duration_limit, OVERLAP_MAX_SECONDS)
            demo_name = "conversation.wav"
        else:
            if model is None:
                model = settings.default_model
            if model not in RECOGNIZERS:
                raise HTTPException(422, "Model must be baseline or cleartalk.")
            models = settings.model_status()
            if mode == "separate" and not models["separator"]["available"]:
                raise HTTPException(503, "The separation model is not installed yet. Complete setup and restart ClearTalk.")
            if not models[model]["available"]:
                if model == "cleartalk" and models["baseline"]["available"]:
                    raise HTTPException(409, "The ClearTalk fine-tuned recognizer is not installed. Select the pretrained model.")
                raise HTTPException(503, "The speech recognizer is not installed yet. Complete setup and restart ClearTalk.")
            upload_limit, duration_limit = settings.max_upload_bytes, settings.max_duration_seconds
            demo_name = "mix.wav"
        job_id = manager.reserve(request.state.owner, mode, model, upload_limit, duration_limit)
        job_dir = settings.jobs_dir / job_id
        uploaded_path = None
        try:
            if demo:
                input_path = (settings.root / "samples/two_voices_kk_ru_en/kazakh_and_russian.wav"
                              if conversation and two_voices else settings.demo_dir / demo_name)
                if not input_path.is_file():
                    raise HTTPException(404, "The demo recording is not installed yet.")
            else:
                extension = Path(file.filename or "").suffix.lower()
                if extension not in ALLOWED_EXTENSIONS:
                    raise HTTPException(422, "Supported files: WAV, MP3, M4A, FLAC, OGG, AAC, MP4 and WebM.")
                uploaded_path = job_dir / f"upload{extension}"
                total = 0
                with uploaded_path.open("wb") as handle:
                    while chunk := await file.read(1024 * 1024):
                        total += len(chunk)
                        if total > upload_limit:
                            raise HTTPException(413, f"Please choose a file smaller than {upload_limit // (1024 * 1024)} MB.")
                        handle.write(chunk)
                input_path = uploaded_path
            audio = await asyncio.to_thread(decode_limited, input_path, settings.sample_rate, duration_limit)
            import soundfile as sf

            await asyncio.to_thread(sf.write, job_dir / "original.wav", audio, settings.sample_rate, subtype="PCM_16")
            context = {"is_demo": demo, "model": model}
            if conversation:
                context.update(language=language, speakers=speakers)
                if two_voices:
                    context["two_voices"] = True
            (job_dir / "input.json").write_text(json.dumps(context), encoding="utf-8")
            return manager.submit(job_id, mode)
        except AudioValidationError as exc:
            manager.discard(job_id)
            raise HTTPException(422, str(exc)) from exc
        except BaseException:
            manager.discard(job_id)
            raise
        finally:
            if uploaded_path is not None:
                uploaded_path.unlink(missing_ok=True)
            if file is not None:
                await file.close()

    def owned_job(job_id: str, request: Request):
        if not JOB_ID.fullmatch(job_id):
            raise HTTPException(404, "This recording was not found or has expired.")
        job = manager.get(job_id, request.state.owner)
        if job is None:
            raise HTTPException(404, "This recording was not found or has expired. Open it in the browser that created it.")
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        return owned_job(job_id, request)

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, request: Request):
        owned_job(job_id, request)
        manager.delete(job_id, request.state.owner)

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str, request: Request):
        job = owned_job(job_id, request)
        if job["status"] != "done":
            raise HTTPException(409, "The transcript is not ready yet.")
        return FileResponse(settings.jobs_dir / job_id / "transcript.txt", media_type="text/plain", filename="cleartalk-transcript.txt")

    @app.post("/api/jobs/{job_id}/example")
    def save_example(job_id: str, payload: CheckedExample, request: Request):
        """Keep a finished conversation with its corrected transcript as a local check example."""
        if settings.deployment != "local":
            raise HTTPException(403, "Check examples can only be kept on a local installation.")
        if not payload.consent:
            raise HTTPException(422, "Confirm that everyone in the recording agreed to it being kept.")
        job = owned_job(job_id, request)
        if job["status"] != "done" or (job.get("result") or {}).get("mode") != "conversation":
            raise HTTPException(409, "Only a finished conversation can be kept as a check example.")
        turns = [turn.model_dump() for turn in payload.turns if turn.text.strip()]
        if not turns or any(turn["end"] < turn["start"] for turn in turns):
            raise HTTPException(422, "The corrected transcript has no text or a turn that ends before it starts.")
        import shutil
        import time as clock

        target = settings.examples_dir / job_id
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(settings.jobs_dir / job_id / "original.wav", target / "audio.wav")
        result = job["result"]
        (target / "reference.json").write_text(json.dumps({
            "id": job_id, "saved_at": clock.strftime("%Y-%m-%d %H:%M"), "duration_seconds": result["duration_seconds"],
            "note": payload.note, "consent_confirmed": True, "turns": turns,
            "first_transcript": {"language": result.get("language"), "model": result.get("model"),
                                 "recognizers": result["models"]["asr"], "turns": result.get("turns", [])},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        count = sum((path / "reference.json").is_file() for path in settings.examples_dir.iterdir() if path.is_dir())
        return {"saved": True, "examples": count}

    @app.get("/media/{job_id}/{filename}")
    def media(job_id: str, filename: str, request: Request):
        owned_job(job_id, request)
        if filename not in MEDIA_FILES:
            raise HTTPException(404, "Audio file not found.")
        target = settings.jobs_dir / job_id / filename
        if not target.is_file():
            raise HTTPException(404, "Audio file not found.")
        return FileResponse(target, media_type="audio/wav")

    @app.get("/")
    def home():
        web = settings.root / "web"
        index = web / "index.html"
        if not index.is_file():
            return PlainTextResponse("ClearTalk backend is running. The web interface has not been installed.")
        # Stamp the script and stylesheet with their modification time, so a browser never
        # pairs a fresh page with a stale cached script after the interface is updated.
        page = index.read_text(encoding="utf-8")
        for name in ("i18n.js", "app.js", "styles.css"):
            asset = web / name
            if asset.is_file():
                page = page.replace(f"/static/{name}", f"/static/{name}?v={int(asset.stat().st_mtime)}")
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    app.mount("/static", FreshStaticFiles(directory=str(settings.root / "web"), check_dir=False), name="static")
    return app


app = create_app()
