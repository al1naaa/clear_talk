"""Write dist/ClearTalk_source.zip: code, frozen plans and small manifests - no audio, weights or user data.

    .venv-train/bin/python scripts/package_source.py          (Windows: .venv-train\\Scripts\\python scripts\\package_source.py)

The archive carries the project between computers and is what
the Colab notebook expects to be uploaded. Model weights travel separately: scripts/package_models.py.
Files are chosen from an explicit
allow-list, so recordings, datasets, virtual environments and secrets can never be included.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "README.md", "README_RU.md", "LICENSE", "requirements.txt", "requirements-train.txt", "pytest.ini",
    "Dockerfile", ".dockerignore", "Start ClearTalk.command", "Start ClearTalk.bat",
    "backend/*.py", "web/*", "cleartalk_asr/*.py", "scripts/*.py", "tests/*.py",
    "ml_pipeline_english/*.py", "ml_pipeline_english/PIPELINE.md", "ml_pipeline_english/data_profile.json",
    "ml_pipeline_english/figures/*.png", "ml_pipeline/*.py", "ml_pipeline/PIPELINE.md", "ml_pipeline/data_profile.json", "ml_pipeline/reports/*",
    "ml_pipeline/figures/*.png",
    "ml_pipeline/.guard_state.json",
    "data/asr/plans/*", "data/asr/results/*.json", "data/v4/esc10/manifest.json", "data/v4/esc10/LICENSE",
    "data/demo/metadata.json", "data/demo/mix.wav", "data/demo/noise.wav", "data/demo/source1.wav", "data/demo/source2.wav",
    "deployment/*", "notebooks/*.ipynb",
    "docs/DETAILS*.md", "docs/DEMO_RU.md", "docs/NOISE_ROADMAP.md", "docs/LOCAL_CHECK_2026-11-10.json", "docs/NOISE_SMOKE_2026-11-10.json",
    # multilingual work: code, frozen recipes, validation/test transcripts and scores, the assembled demo
    "cleartalk_ml/*.py", "data/ml/plans/*", "data/ml/results/*.json", "data/ml/results/hyp/*.jsonl",
    "data/ml/results/hyp-test/*.jsonl", "data/ml/downloads.json", "data/demo/conversation.wav", "data/demo/conversation.json", "data/demo/conversation_cafe.wav",
    # round 2 (the large model): its own log, notebooks, figures, frozen test B and the transcripts of its final test
    "scripts/*.sh", "ml_pipeline/round2/*.py", "ml_pipeline/round2/PIPELINE.md", "ml_pipeline/round2/figures/*.png",
    "ml_pipeline/round2/.guard_state.json", "data/ml/results/hyp-testB/*.jsonl",
    # round 3 (closed without training) and round 4 (noise, protections against invented text, test D)
    "ml_pipeline/round3/*.py", "ml_pipeline/round3/PIPELINE.md", "ml_pipeline/round3/.guard_state.json",
    "ml_pipeline/round4/*.py", "ml_pipeline/round4/PIPELINE.md", "ml_pipeline/round4/figures/*.png", "ml_pipeline/round4/reports/*.py",
    "ml_pipeline/round4/.guard_state.json", "data/ml/results/hyp-testD/*.jsonl", "data/ml/results/hyp-robust/*.jsonl",
    "data/ml/results/hyp-overlap/*.jsonl",
    "ml_pipeline/noise_followup_20261110/*.py", "ml_pipeline/noise_followup_20261110/*.json",
    "ml_pipeline/noise_followup_20261110/PIPELINE.md", "ml_pipeline/noise_followup_20261110/figures/*.png",
    "data/ml/noise_followup_20261110/*.csv", "data/ml/noise_followup_20261110/*.jsonl",
    # Public documentation, README images and attributed two-voice examples.
    "docs/assets/*", ".gitignore", ".gitattributes", ".github/workflows/*", "samples/two_voices_kk_ru_en/*",
]
# data/real/ (recordings a user kept as check examples) is private and is deliberately not in this list.
EXCLUDE_PARTS = {".venv", ".venv-train", "__pycache__", "models", "jobs", "runtime", "dev"}


def files():
    seen = set()
    for pattern in INCLUDE:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and not path.is_symlink() and not EXCLUDE_PARTS & set(path.relative_to(ROOT).parts) and path not in seen:
                seen.add(path)
                yield path


def main():
    out = ROOT / "dist" / "ClearTalk_source.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        count = 0
        for path in files():
            archive.write(path, path.relative_to(ROOT).as_posix())
            count += 1
    print(f"{out} · {count} files · {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
