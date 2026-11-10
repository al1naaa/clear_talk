"""Download the two public, revision-pinned pretrained models ClearTalk builds on.

The ClearTalk fine-tuned recognizer is not downloaded: it is produced locally by
``python -m cleartalk_asr.export`` after training (see README).
"""
from pathlib import Path
import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("speechbrain/sepformer-libri2mix", "eb43c5bfbb2aa654630adbf849373bcec0a20ed4",
     "candidates-v3/sepformer-libri2mix", ["hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt", "README.md"]),
    ("Systran/faster-whisper-small.en", "d1d751a5f8271d482d14ca55d9e2deeebbae577f",
     "candidates-v3/whisper-small.en", ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt", "README.md"]),
]
# Needed only to train or re-export the fine-tuned model (PyTorch weights + tokenizer files).
TRAINING_MODELS = [
    ("openai/whisper-small.en", None, "whisper-small.en-hf", ["*.json", "*.txt", "model.safetensors"]),
]


def selected_models(include_training=False):
    return MODELS + (TRAINING_MODELS if include_training else [])


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="also download openai/whisper-small.en for fine-tuning")
    args = parser.parse_args()
    models_dir = Path(os.getenv("CLEARTALK_MODELS_DIR") or ROOT / "models").expanduser()
    for repo_id, revision, folder, files in selected_models(args.training):
        print(f"Preparing {repo_id} ...", flush=True)
        snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=files,
                          local_dir=models_dir / folder, max_workers=2)
    print("ClearTalk models are ready. Audio processing runs locally.", flush=True)


if __name__ == "__main__":
    main()
