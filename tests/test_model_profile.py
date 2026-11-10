"""The model profile fixes which weights the service may load."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from backend.config import DEFAULT_PROFILE, Settings

CODE_ROOT = Path(__file__).resolve().parents[1]


def write_profile(tmp_path, **overrides):
    value = copy.deepcopy(DEFAULT_PROFILE)
    value.update(overrides)
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models/active_profile.json").write_text(json.dumps(value))
    return value


def install(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(name.encode())
    return {name: hashlib.sha256(name.encode()).hexdigest() for name in names}


def test_profile_resolves_directories_and_is_fixed_per_process(tmp_path):
    write_profile(tmp_path)
    settings = Settings(root=tmp_path)
    assert settings.separator_id == "speechbrain/sepformer-libri2mix"
    assert settings.recognizer_dir("baseline") == tmp_path / "models/candidates-v3/whisper-small.en"
    assert settings.recognizer_dir("cleartalk") == tmp_path / "models/whisper-small.en-cleartalk"
    write_profile(tmp_path, default_model="cleartalk")
    assert settings.default_model == "baseline"


@pytest.mark.parametrize("change", [
    {"version": 4}, {"default_model": "custom"}, {"asr_policy": "unverified"},
    {"recognizers": {"baseline": {"name": "x", "directory": "../secret"}, "cleartalk": {"name": "y", "directory": "ok"}}},
    {"recognizers": {"baseline": {"name": "x", "directory": "a"}}},
    {"separator": {"id": "x", "directory": "/absolute"}},
])
def test_invalid_profile_is_rejected(tmp_path, change):
    write_profile(tmp_path, **change)
    with pytest.raises(ValueError, match="Invalid active"):
        Settings(root=tmp_path).model_status()


def test_signatures_are_checked_before_loading(tmp_path):
    profile = write_profile(tmp_path)
    settings = Settings(root=tmp_path)
    signatures = install(settings.recognizer_dir("cleartalk"), ("model.bin", "config.json", "tokenizer.json"))
    profile["recognizers"]["cleartalk"]["files_sha256"] = signatures
    write_profile(tmp_path, recognizers=profile["recognizers"])
    settings = Settings(root=tmp_path)
    settings.verify_recognizer("cleartalk")
    (settings.recognizer_dir("cleartalk") / "model.bin").write_bytes(b"different weights")
    with pytest.raises(RuntimeError, match="differs from the validated profile"):
        settings.verify_recognizer("cleartalk")
    assert settings.model_status()["cleartalk"]["available"]  # installed, but refused at load time


def test_transcription_policy_drift_is_rejected(tmp_path):
    write_profile(tmp_path, transcription_policy_sha256="wrong")
    settings = Settings(root=tmp_path)
    install(settings.recognizer_dir("baseline"), ("model.bin", "config.json", "tokenizer.json"))
    with pytest.raises(RuntimeError, match="transcription policy differs"):
        settings.verify_recognizer("baseline")


def test_missing_profile_uses_safe_defaults(tmp_path):
    settings = Settings(root=tmp_path)
    assert settings.default_model == "baseline"
    status = settings.model_status()
    assert not status["separator"]["available"] and not status["cleartalk"]["available"]


def test_repository_profile_matches_installed_weights_if_present():
    path = CODE_ROOT / "models/active_profile.json"
    if not path.is_file():
        pytest.skip("models are not installed in this checkout")
    settings = Settings()
    settings.verify_separator()
    settings.verify_recognizer("baseline")
    if settings.model_status()["cleartalk"]["available"]:
        settings.verify_recognizer("cleartalk")


def test_downloader_lists_pinned_pretrained_models():
    from scripts.download_models import selected_models

    names = [entry[0] for entry in selected_models()]
    assert names == ["speechbrain/sepformer-libri2mix", "Systran/faster-whisper-small.en"]
    assert all(entry[1] for entry in selected_models())
