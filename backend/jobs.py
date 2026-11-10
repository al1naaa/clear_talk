"""Single-worker jobs with bounded storage, ownership and restart recovery."""

import copy
import json
import logging
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException

logger = logging.getLogger(__name__)
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
ACTIVE = {"uploading", "queued", "running"}


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02d}"


def conversation_text(result: dict) -> str:
    """Plain-text transcript of a conversation: one line per turn, with time, speaker and language."""
    sections = ["CLEARTALK - TRANSCRIPT", f"Audio duration: {clock(result['duration_seconds'])}", "Mode: conversation",
                f"Speech recognizer: {result['models']['asr']}"]
    if result["models"].get("speakers"):
        sections.append(f"Speaker labels: {result['models']['speakers']}")
    sections.extend(["", "CONVERSATION", "------------"])
    if not result["turns"]:
        sections.append("No speech transcribed.")
    for turn in result["turns"]:
        flag = "  [check]" if turn.get("needs_review") else ""
        sections.append(f"[{clock(turn['start'])}-{clock(turn['end'])}] {turn['speaker']} ({turn['language_name']}): {turn['text']}{flag}")
    sections.extend(["", "LIMITATIONS"])
    sections.extend(f"• {warning}" for warning in result.get("warnings", []))
    return "\n".join(sections) + "\n"


def transcript_text(result: dict) -> str:
    if result.get("mode") == "conversation":
        return conversation_text(result)
    sections = ["CLEARTALK - TRANSCRIPT", f"Audio duration: {result['duration_seconds']:.2f} seconds", f"Mode: {result['mode']}", f"Speech recognizer: {result['models']['asr']}"]
    tracks = [("Original recording", result["original"])]
    tracks.extend((source["label"], source) for source in result["sources"])
    for title, track in tracks:
        sections.extend(["", title.upper(), "-" * len(title)])
        if track["segments"]:
            sections.extend(f"[{segment['start']:.2f}-{segment['end']:.2f}s] {segment['text']}" for segment in track["segments"])
        else:
            sections.append(track.get("text") or "No speech transcribed.")
        for note in track.get("review_notes", []):
            sections.append(f"Review note: {note}")
    sections.extend(["", "LIMITATIONS"])
    sections.extend(f"• {warning}" for warning in result.get("warnings", []))
    return "\n".join(sections) + "\n"


class JobManager:
    def __init__(self, settings, processor):
        self.settings = settings
        self.processor = processor
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cleartalk")
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.started = False

    def start(self):
        # Importing backend.app (for tests or utilities) must not rewrite the
        # running server's persisted jobs. Restore only at service startup.
        with self.lock:
            if not self.started:
                self._restore()
                self.cleanup()
                self.started = True

    def _persist(self, job_id):
        directory = self.settings.jobs_dir / job_id
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(self.jobs[job_id], ensure_ascii=False), encoding="utf-8")
        temporary.replace(directory / "job.json")

    def _restore(self):
        for directory in self.settings.jobs_dir.iterdir():
            if directory.is_symlink() or not directory.is_dir() or not JOB_ID.fullmatch(directory.name):
                continue
            try:
                job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
                if not isinstance(job, dict):
                    continue
                if job.get("id") != directory.name or not re.fullmatch(r"[0-9a-f]{64}", job.get("owner", "")):
                    continue
                if not isinstance(job.get("expires_at"), (int, float)):
                    continue
                if job.get("status") in ACTIVE:
                    job.update(status="error", stage="Processing interrupted", error="The service restarted before this recording finished. Please upload it again.", error_code="worker_restarted")
                if job.get("status") not in {"done", "error"}:
                    continue
                self.jobs[directory.name] = job
                self._persist(directory.name)
            except (OSError, ValueError, TypeError):
                # Legacy v1 recordings have no ownership metadata. They are
                # never attached to whichever browser opens the new server.
                continue

    def cleanup(self):
        with self.lock:
            expired = [job_id for job_id, job in self.jobs.items() if job["status"] not in ACTIVE and job["expires_at"] <= time.time()]
            for job_id in expired:
                directory = self.settings.jobs_dir / job_id
                if directory.is_dir() and not directory.is_symlink():
                    shutil.rmtree(directory)
                self.jobs.pop(job_id, None)

    def _storage_bytes(self):
        # Count all retained recordings, including unmigrated v1 directories.
        total = 0
        for path in self.settings.jobs_dir.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                # Another upload may finish and remove its raw input here.
                continue
        return total

    def pending(self):
        with self.lock:
            return sum(job["status"] in ACTIVE for job in self.jobs.values())

    def reserve(self, owner: str, mode: str, model: str, upload_bytes: int | None = None, seconds: float | None = None):
        """Reserve a job; ``upload_bytes`` / ``seconds`` are the limits of this mode (defaults: the short modes)."""
        upload_bytes = upload_bytes or self.settings.max_upload_bytes
        seconds = seconds or self.settings.max_duration_seconds
        with self.lock:
            self.cleanup()
            pending = self.pending()
            if pending >= self.settings.max_pending_jobs:
                raise HTTPException(429, "The processing queue is full. Please wait for a recording to finish.")
            stored_count = sum(path.is_dir() and not path.is_symlink() for path in self.settings.jobs_dir.iterdir())
            if stored_count >= self.settings.max_stored_jobs:
                raise HTTPException(507, "Recording storage is full. Delete an old result or wait for its retention period to expire.")
            # Reserve worst-case input/output space for every concurrent upload.
            reservation = upload_bytes + int(seconds * self.settings.sample_rate * 8) + 1024 * 1024
            if self._storage_bytes() + (pending + 1) * reservation > self.settings.max_storage_bytes:
                raise HTTPException(507, "Recording storage is full. Delete an old result or wait for its retention period to expire.")
            job_id = uuid.uuid4().hex
            (self.settings.jobs_dir / job_id).mkdir(mode=0o700)
            now = time.time()
            self.jobs[job_id] = {"id": job_id, "owner": owner, "status": "uploading", "stage": "Validating the recording", "progress": 0, "mode": mode, "model": model, "created_at": now, "expires_at": now + self.settings.job_ttl_seconds}
            self._persist(job_id)
            return job_id

    def get(self, job_id: str, owner: str):
        self.cleanup()
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.get("owner") != owner:
                return None
            return copy.deepcopy({key: value for key, value in job.items() if key != "owner"})

    def discard(self, job_id: str):
        with self.lock:
            self.jobs.pop(job_id, None)
            directory = self.settings.jobs_dir / job_id
            if directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory)

    def delete(self, job_id: str, owner: str):
        with self.lock:
            job = self.get(job_id, owner)
            if job is None:
                raise HTTPException(404, "This recording was not found or has expired.")
            if job["status"] in ACTIVE:
                raise HTTPException(409, "Please wait for processing to finish before deleting this recording.")
            self.discard(job_id)

    def submit(self, job_id: str, mode: str):
        with self.lock:
            self.jobs[job_id].update(status="queued", stage="Waiting for the worker")
            self._persist(job_id)
            response = self.get(job_id, self.jobs[job_id]["owner"])
            self.executor.submit(self._run, job_id, mode)
            return response

    def _run(self, job_id: str, mode: str):
        def update(stage: str, progress: int):
            with self.lock:
                self.jobs[job_id].update(status="running", stage=stage, progress=progress)
                self._persist(job_id)

        update("Preparing the recording", 5)
        directory = self.settings.jobs_dir / job_id
        try:
            result = self.processor.process(job_id, directory, mode, update)
            (directory / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            (directory / "transcript.txt").write_text(transcript_text(result), encoding="utf-8")
            with self.lock:
                self.jobs[job_id].update(status="done", stage="Complete", progress=100, result=result, expires_at=time.time() + self.settings.job_ttl_seconds)
                self._persist(job_id)
        except Exception as exc:
            logger.exception("Audio processing failed for job %s", job_id)
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                message, code = "There was not enough memory. Try a shorter recording.", "insufficient_memory"
            elif "missing model files" in str(exc).lower() or "differs from the validated profile" in str(exc).lower():
                message, code = "A speech model this recording needs is not installed correctly. Complete setup and restart ClearTalk.", "models_unavailable"
            else:
                message, code = "Audio processing failed. Try a shorter WAV recording. Technical details are available in the service logs.", "inference_failed"
            with self.lock:
                self.jobs[job_id].update(status="error", stage="Processing stopped", error=message, error_code=code, expires_at=time.time() + self.settings.job_ttl_seconds)
                self._persist(job_id)
