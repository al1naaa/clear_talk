"""Start the single-worker hosted app; download revision-pinned public weights."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from backend.config import Settings
    from huggingface_hub import snapshot_download
    from scripts.download_models import selected_models

    settings = Settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    model_root = settings.models_dir
    model_root.mkdir(parents=True, exist_ok=True)
    for repo_id, revision, folder, filenames in selected_models():
        destination = model_root / folder
        required = [name for name in filenames if name != "README.md"]
        if all((destination / name).is_file() for name in required):
            continue
        print(f"Downloading pinned inference model: {repo_id}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=filenames,
            local_dir=destination,
            max_workers=2,
        )
    if settings.default_model == "cleartalk" and not settings.model_status()["cleartalk"]["available"]:
        raise SystemExit("The ClearTalk fine-tuned recognizer is missing from this image; export it before building.")

    port = int(os.environ.get("PORT", "7860"))
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be in the range 1-65535.")
    print("ClearTalk is starting with one bounded inference worker.", flush=True)
    os.execv(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", str(port), "--workers", "1"],
    )


if __name__ == "__main__":
    main()
