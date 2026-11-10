"""Project locations for the multilingual work (the English pipeline keeps cleartalk_asr.paths)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml"
RAW = DATA / "raw"            # downloaded archives, exactly as published
AUDIO = DATA / "audio"        # unpacked audio
PLANS = DATA / "plans"        # frozen, seeded recipes (small, versioned with the code)
RESULTS = DATA / "results"    # hypotheses and scores (small, versioned with the code)
MODELS = Path(os.getenv("CLEARTALK_MODELS_DIR") or ROOT / "models").expanduser() / "multilingual"
RUNS = MODELS / "runs"
PIPELINE = ROOT / "ml_pipeline"

LANGUAGES = ("kk", "ru", "en")
LANGUAGE_NAMES = {"kk": "Kazakh", "ru": "Russian", "en": "English"}
SAMPLE_RATE = 16000

# Ready recognizers: PyTorch folder (for conversion and training) -> CTranslate2 folder (for the app).
READY_MODELS = {
    "turbo": {"name": "Whisper large-v3-turbo", "hf": "whisper-large-v3-turbo-hf", "ct2": "whisper-large-v3-turbo"},
    "kazakh": {"name": "Kazakh Whisper large-v3-turbo", "hf": "kazakh-whisper-large-v3-turbo-hf",
               "ct2": "kazakh-whisper-large-v3-turbo", "tokenizer": "whisper-large-v3-turbo-hf"},
    "small": {"name": "Whisper small (multilingual)", "hf": "whisper-small-hf", "ct2": "whisper-small"},
}
BASE_WHISPER = MODELS / READY_MODELS["small"]["hf"]

# Base models a LoRA can be trained on. ``processor`` is the folder whose feature extractor, tokenizer and
# generation settings are used (the Kazakh fine-tune's own files were written by a newer library; its
# vocabulary is identical to large-v3-turbo's). ``lora`` says where the adapters go: everywhere, or only in
# the decoder with the encoder frozen (what fits 6 GB for a large model).
BASES = {
    "small": {"hf": "whisper-small-hf", "processor": "whisper-small-hf", "lora": "all", "micro_batch": 2, "accumulate": 8,
              "export": "whisper-small-cleartalk-multilingual", "name": "openai/whisper-small"},
    "kazakh-turbo": {"hf": "kazakh-whisper-large-v3-turbo-hf", "processor": "whisper-large-v3-turbo-hf", "lora": "decoder",
                     "micro_batch": 4, "accumulate": 4, "export": "whisper-large-cleartalk-multilingual",
                     "name": "shyngys879/kazakh-whisper-large-v3-turbo"},
}


def base_name() -> str:
    """The base model of the current process (an environment variable, so data-loading workers inherit it)."""
    name = os.environ.get("CLEARTALK_BASE", "small")
    if name not in BASES:
        raise ValueError(f"Unknown base model {name!r}")
    return name
SPEAKER_MODEL = MODELS / "spkrec-ecapa-voxceleb"


def rel(path) -> str:
    """Store paths relative to the project so manifests work on Windows and macOS alike."""
    return Path(path).resolve().relative_to(ROOT).as_posix()
