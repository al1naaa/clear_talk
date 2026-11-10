"""Service settings and the on-disk model profile.

``models/active_profile.json`` names the separator and the two speech recognizers
(pretrained baseline and ClearTalk's fine-tuned model), their SHA-256 file
signatures and the default. The process reads it once and refuses weights that do
not match their signature, so an evaluated model is never silently swapped.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path, PurePosixPath, PureWindowsPath

RECOGNIZERS = ("baseline", "cleartalk")
LANGUAGES = ("kk", "ru", "en")
# Conversation mode: ready multilingual recognizers plus this project's own multilingual model.
# "routes" says which recognizer transcribes which language when the user leaves the choice to the app;
# scripts/activate_multilingual.py rewrites it from validation results.
DEFAULT_MULTILINGUAL_PROFILE = {
    "version": 1,
    "recognizers": {
        "turbo": {"name": "Whisper large-v3-turbo", "directory": "whisper-large-v3-turbo", "languages": ["kk", "ru", "en"]},
        "kazakh": {"name": "Kazakh Whisper large-v3-turbo", "directory": "kazakh-whisper-large-v3-turbo", "languages": ["kk"]},
        "small": {"name": "Whisper small (multilingual)", "directory": "whisper-small", "languages": ["kk", "ru", "en"]},
        "cleartalk": {"name": "ClearTalk multilingual (LoRA fine-tuned)", "directory": "whisper-small-cleartalk-multilingual",
                      "languages": ["kk", "ru", "en"]},
        "cleartalk-large": {"name": "ClearTalk large multilingual (LoRA fine-tuned)",
                            "directory": "whisper-large-cleartalk-multilingual", "languages": ["kk", "ru", "en"]},
        "cleartalk-kk": {"name": "ClearTalk multilingual with Kazakh broadcast (LoRA fine-tuned)",
                         "directory": "whisper-small-cleartalk-kazakh", "languages": ["kk", "ru", "en"]},
    },
    "routes": {"kk": "kazakh", "ru": "turbo", "en": "turbo"},
    "detector": "turbo",
    # The recognizer that stands in when a routed model or the detector is not installed, so the mode also
    # works on a computer that carries only a small model. It must cover all three languages.
    "small_recognizer": "small",
    # The recognizer that writes every turn when the user says the languages change inside turns: the one with
    # the lowest one-pass error on mixed validation recordings (scripts/select_round2.py rewrites it).
    "mixed_recognizer": "cleartalk",
    "speaker_model": {"id": "speechbrain/spkrec-ecapa-voxceleb", "directory": "spkrec-ecapa-voxceleb"},
    "asr_policy": "multilingual-vad",
}
SPEAKER_FILES = ("hyperparams.yaml", "embedding_model.ckpt")
SEPARATOR_FILES = ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt")
RECOGNIZER_FILES = ("model.bin", "config.json", "tokenizer.json")
TRANSCRIPTION_POLICY = Path(__file__).with_name("transcription.py")

DEFAULT_PROFILE = {
    "version": 5,
    "separator": {"id": "speechbrain/sepformer-libri2mix", "directory": "candidates-v3/sepformer-libri2mix"},
    "recognizers": {
        "baseline": {"name": "Whisper small.en (pretrained)", "directory": "candidates-v3/whisper-small.en"},
        "cleartalk": {"name": "ClearTalk Whisper small.en (LoRA fine-tuned)", "directory": "whisper-small.en-cleartalk"},
    },
    "default_model": "baseline",
    "asr_policy": "english-vad-v1",
}


def default_cpu_threads() -> int:
    """Threads for the speech models on the processor when CLEARTALK_CPU_THREADS is not set.

    A Mac runs Conversation mode on its processor (faster-whisper cannot use Apple graphics), so a local
    installation uses most of the cores: all but two, between 2 and 8. The container image sets 2 itself.
    """
    return max(2, min(8, (os.cpu_count() or 4) - 2))


def _optional_path(name: str):
    value = os.getenv(name)
    return Path(value).expanduser() if value else None


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _safe_directory(value) -> bool:
    """A relative folder name without parent references - judged by both POSIX and Windows rules,
    so a profile written on one system cannot point outside the models folder on the other."""
    if not isinstance(value, str) or not value:
        return False
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    return not (posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root
                or ".." in posix.parts or ".." in windows.parts)


@dataclass(frozen=True)
class Settings:
    root: Path = Path(__file__).resolve().parents[1]
    max_duration_seconds: float = field(default_factory=lambda: float(os.getenv("CLEARTALK_MAX_DURATION_SECONDS", "20")))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("CLEARTALK_MAX_UPLOAD_MB", "20")) * 1024 * 1024)
    max_pending_jobs: int = field(default_factory=lambda: int(os.getenv("CLEARTALK_MAX_PENDING_JOBS", "5")))
    sample_rate: int = 16000
    cpu_threads: int = field(default_factory=lambda: int(os.getenv("CLEARTALK_CPU_THREADS") or default_cpu_threads()))
    data_path: Path | None = field(default_factory=lambda: _optional_path("CLEARTALK_DATA_DIR"))
    models_path: Path | None = field(default_factory=lambda: _optional_path("CLEARTALK_MODELS_DIR"))
    job_ttl_seconds: float = field(default_factory=lambda: float(os.getenv("CLEARTALK_JOB_TTL_HOURS", "24")) * 3600)
    max_storage_bytes: int = field(default_factory=lambda: int(os.getenv("CLEARTALK_MAX_STORAGE_MB", "1024")) * 1024 * 1024)
    max_stored_jobs: int = field(default_factory=lambda: int(os.getenv("CLEARTALK_MAX_STORED_JOBS", "200")))
    access_token: str = field(default_factory=lambda: os.getenv("CLEARTALK_ACCESS_TOKEN", ""), repr=False)
    session_secret: str = field(default_factory=lambda: os.getenv("CLEARTALK_SESSION_SECRET", ""), repr=False)
    secure_cookies: bool = field(default_factory=lambda: os.getenv("CLEARTALK_SECURE_COOKIES", "false").lower() in {"1", "true", "yes"})
    deployment: str = field(default_factory=lambda: os.getenv("CLEARTALK_DEPLOYMENT", "local"))
    # Conversation mode (multilingual, any number of speakers) accepts long recordings.
    max_long_seconds: float = field(default_factory=lambda: float(os.getenv("CLEARTALK_MAX_LONG_MINUTES", "60")) * 60)
    max_long_upload_bytes: int = field(default_factory=lambda: int(os.getenv("CLEARTALK_MAX_LONG_UPLOAD_MB", "300")) * 1024 * 1024)
    device: str = field(default_factory=lambda: os.getenv("CLEARTALK_DEVICE", "auto").lower())

    def __post_init__(self):
        for name in ("max_duration_seconds", "max_upload_bytes", "max_pending_jobs", "sample_rate", "cpu_threads", "job_ttl_seconds", "max_storage_bytes", "max_stored_jobs", "max_long_seconds", "max_long_upload_bytes"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_duration_seconds > 60:
            raise ValueError("This inference service supports at most 60 seconds per recording")
        if self.max_long_seconds > 4 * 3600:
            raise ValueError("Conversation mode supports at most four hours per recording")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("CLEARTALK_DEVICE must be auto, cpu or cuda")

    @property
    def data_dir(self) -> Path:
        return self.data_path or self.root / "data"

    @property
    def models_dir(self) -> Path:
        return self.models_path or self.root / "models"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def demo_dir(self) -> Path:
        return self.root / "data" / "demo"

    @property
    def examples_dir(self) -> Path:
        """Recordings whose transcript a person corrected and chose to keep as a check example.

        They stay on this computer: the folder is outside version control and outside every archive
        the project writes. cleartalk_ml.real scores the recognizers on them."""
        return self.root / "data" / "real"

    @cached_property
    def active_profile(self) -> dict:
        """Read once per process; an invalid profile stops the service instead of guessing."""
        path = self.models_dir / "active_profile.json"
        profile = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else DEFAULT_PROFILE
        recognizers = profile.get("recognizers")
        if (profile.get("version") != 5 or not isinstance(profile.get("separator"), dict)
                or not _safe_directory(profile["separator"].get("directory"))
                or not isinstance(recognizers, dict) or set(recognizers) != set(RECOGNIZERS)
                or not all(_safe_directory(r.get("directory")) for r in recognizers.values())
                or profile.get("default_model") not in RECOGNIZERS
                or profile.get("asr_policy") != "english-vad-v1"):
            raise ValueError("Invalid active model profile; refusing an unverified model combination")
        return profile

    @property
    def default_model(self) -> str:
        return self.active_profile["default_model"]

    @property
    def separator_id(self) -> str:
        return self.active_profile["separator"]["id"]

    @property
    def separator_dir(self) -> Path:
        return self.models_dir / self.active_profile["separator"]["directory"]

    def recognizer_dir(self, model: str) -> Path:
        return self.models_dir / self.active_profile["recognizers"][model]["directory"]

    def recognizer_name(self, model: str) -> str:
        return self.active_profile["recognizers"][model]["name"]

    def _check_files(self, directory: Path, required, signatures) -> None:
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing model files in {directory.name}: {', '.join(missing)}")
        for name, expected in (signatures or {}).items():
            if sha256(directory / name) != expected:
                raise RuntimeError(f"The file {directory.name}/{name} differs from the validated profile")

    def verify_separator(self) -> None:
        separator = self.active_profile["separator"]
        self._check_files(self.separator_dir, SEPARATOR_FILES, separator.get("files_sha256"))

    def verify_recognizer(self, model: str) -> None:
        """Check files and signatures before the first load of a recognizer."""
        recognizer = self.active_profile["recognizers"][model]
        expected_policy = self.active_profile.get("transcription_policy_sha256")
        if expected_policy and sha256(TRANSCRIPTION_POLICY) != expected_policy:
            raise RuntimeError("The transcription policy differs from the validated profile")
        self._check_files(self.recognizer_dir(model), RECOGNIZER_FILES, recognizer.get("files_sha256"))

    def model_card(self, model: str) -> dict | None:
        path = self.recognizer_dir(model) / "model_card.json"
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return card if isinstance(card, dict) else None

    def recognizer_status(self, model: str) -> dict:
        """Cheap installation check for status pages; signatures are verified before loading."""
        directory = self.recognizer_dir(model)
        status = {"name": self.recognizer_name(model), "default": model == self.default_model,
                  "available": all((directory / name).is_file() for name in RECOGNIZER_FILES)}
        if model == "cleartalk":
            status["model_card"] = self.model_card(model)
        return status

    # ------------------------------------------------------------------ conversation mode (multilingual)
    @property
    def multilingual_dir(self) -> Path:
        return self.models_dir / "multilingual"

    @cached_property
    def multilingual_profile(self) -> dict:
        """Read once per process; an invalid profile stops the service instead of guessing."""
        path = self.multilingual_dir / "profile.json"
        profile = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else DEFAULT_MULTILINGUAL_PROFILE
        recognizers = profile.get("recognizers")
        if (profile.get("version") != 1 or not isinstance(recognizers, dict) or not recognizers
                or not all(isinstance(r, dict) and _safe_directory(r.get("directory")) and isinstance(r.get("name"), str)
                           and set(r.get("languages") or ()) <= set(LANGUAGES) and r.get("languages")
                           for r in recognizers.values())
                or set(profile.get("routes") or ()) != set(LANGUAGES)
                or not all(profile["routes"][lang] in recognizers and lang in recognizers[profile["routes"][lang]]["languages"]
                           for lang in LANGUAGES)
                or profile.get("detector") not in recognizers
                or profile.get("small_recognizer", "small") not in recognizers
                or set(recognizers[profile.get("small_recognizer", "small")]["languages"]) != set(LANGUAGES)
                or profile.get("mixed_recognizer", profile.get("small_recognizer", "small")) not in recognizers
                or set(recognizers[profile.get("mixed_recognizer", profile.get("small_recognizer", "small"))]["languages"]) != set(LANGUAGES)
                or not isinstance(profile.get("speaker_model"), dict)
                or not _safe_directory(profile["speaker_model"].get("directory"))
                or profile.get("asr_policy") != "multilingual-vad"):
            raise ValueError("Invalid multilingual model profile; refusing an unverified model combination")
        return profile

    def evaluation_override(self, key: str) -> Path | None:
        """Measurement only: CLEARTALK_EVAL_OVERRIDE="key=folder;…" makes a recognizer load from another folder under
        the multilingual models (to measure the app with an earlier model). The app itself never sets it."""
        for item in os.environ.get("CLEARTALK_EVAL_OVERRIDE", "").split(";"):
            name, _, folder = item.partition("=")
            if name.strip() == key and folder.strip():
                return self.multilingual_dir / folder.strip()
        return None

    def multilingual_recognizer_dir(self, key: str) -> Path:
        return self.evaluation_override(key) or self.multilingual_dir / self.multilingual_profile["recognizers"][key]["directory"]

    @property
    def speaker_model_dir(self) -> Path:
        return self.multilingual_dir / self.multilingual_profile["speaker_model"]["directory"]

    def verify_multilingual_recognizer(self, key: str) -> None:
        recognizer = self.multilingual_profile["recognizers"][key]
        signatures = None if self.evaluation_override(key) else recognizer.get("files_sha256")   # an earlier model has its own
        self._check_files(self.multilingual_recognizer_dir(key), RECOGNIZER_FILES, signatures)

    def multilingual_card(self, key: str) -> dict | None:
        try:
            card = json.loads((self.multilingual_recognizer_dir(key) / "model_card.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return card if isinstance(card, dict) else None

    def multilingual_status(self) -> dict:
        """Cheap installation check of the conversation mode for status pages."""
        profile = self.multilingual_profile
        recognizers = {}
        for key, entry in profile["recognizers"].items():
            directory = self.multilingual_recognizer_dir(key)
            recognizers[key] = {"name": entry["name"], "languages": list(entry["languages"]),
                                "available": all((directory / name).is_file() for name in RECOGNIZER_FILES)}
            if key.startswith("cleartalk"):
                recognizers[key]["model_card"] = self.multilingual_card(key)
        small = profile.get("small_recognizer", "small")

        def installed(key):  # the profile's choice, or the small recognizer standing in for a missing model
            return key if recognizers[key]["available"] or not recognizers[small]["available"] else small

        routes = {language: installed(profile["routes"][language]) for language in LANGUAGES}
        detector = installed(profile["detector"])
        mixed = installed(profile.get("mixed_recognizer", small))
        speakers = all((self.speaker_model_dir / name).is_file() for name in SPEAKER_FILES)
        return {"recognizers": recognizers, "routes": routes, "detector": detector,
                "preferred_routes": dict(profile["routes"]), "small_recognizer": small, "mixed_recognizer": mixed,
                "mixed_available": recognizers[mixed]["available"] and recognizers[detector]["available"],
                "speaker_labels": speakers,
                "available": recognizers[detector]["available"]
                and all(recognizers[routes[lang]]["available"] for lang in LANGUAGES)}

    def model_status(self) -> dict:
        separator_ok = all((self.separator_dir / name).is_file() for name in SEPARATOR_FILES)
        return {
            "separator": {"name": f"{self.separator_id} · 8 kHz", "available": separator_ok},
            **{model: self.recognizer_status(model) for model in RECOGNIZERS},
        }
