"""Write dist/ClearTalk_multilingual_models.zip: the model folders Conversation mode needs on another computer.

    python scripts/package_models.py              # what the app needs
    python scripts/package_models.py --training   # also the LoRA adapter and the PyTorch base model (to retrain)
    python scripts/package_models.py --english    # the English two-voice mode: its models and demo audio (run it on
                                                  # the Mac, where the fine-tuned English recognizer lives)

Inside the archive every path starts with ``models/multilingual/``, so unpacking it in the project folder
puts everything in place. The folders are plain files (CTranslate2 models, a SpeechBrain model, JSON) and
work unchanged on Windows and macOS. Nothing is compressed - weights do not shrink - so it is fast.
"""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "multilingual"
APP_FOLDERS = ["whisper-small-cleartalk-multilingual", "whisper-large-cleartalk-multilingual", "whisper-large-v3-turbo",
               "kazakh-whisper-large-v3-turbo", "whisper-small", "spkrec-ecapa-voxceleb"]
SPEAKER_FILES = {"hyperparams.yaml", "embedding_model.ckpt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--english", action="store_true", help="write ClearTalk_english_models.zip instead")
    args = parser.parse_args()
    if args.english:
        return english()
    paths = [MODELS / "profile.json"] if (MODELS / "profile.json").is_file() else []
    for folder in APP_FOLDERS:
        for path in sorted((MODELS / folder).glob("*")):
            if path.is_file() and (folder != "spkrec-ecapa-voxceleb" or path.name in SPEAKER_FILES):
                paths.append(path)
    if args.training:
        selection = ROOT / "data" / "ml" / "results" / "selection.json"
        if selection.is_file():
            chosen = json.loads(selection.read_text(encoding="utf-8"))
            adapter = MODELS / "runs" / chosen["run"] / f"step-{chosen['chosen_step']:05d}"
            paths += sorted(p for p in adapter.glob("*") if p.is_file())
            paths += sorted(p for p in (MODELS / "runs" / chosen["run"]).glob("*.json*") if p.is_file())
            paths += sorted(p for p in (MODELS / "runs").glob("*.json") if p.is_file())
        paths += sorted(p for p in (MODELS / "whisper-small-hf").glob("*") if p.is_file())
        second = ROOT / "data" / "ml" / "results" / "selection-round2.json"   # the large model of round 2
        if second.is_file():
            chosen = json.loads(second.read_text(encoding="utf-8"))
            adapter = MODELS / "runs" / chosen["run"] / f"step-{chosen['chosen_step']:05d}"
            paths += sorted(p for p in adapter.glob("*") if p.is_file())
            paths += sorted(p for p in (MODELS / "runs" / chosen["run"]).glob("*.json*") if p.is_file())
            for folder in ("kazakh-whisper-large-v3-turbo-hf", "whisper-large-v3-turbo-hf"):
                paths += sorted(p for p in (MODELS / folder).glob("*") if p.is_file())
        paths = list(dict.fromkeys(paths))
    out = ROOT / "dist" / ("ClearTalk_multilingual_models_with_training.zip" if args.training else "ClearTalk_multilingual_models.zip")
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"{out} · {len(paths)} files · {out.stat().st_size / 1e9:.2f} GB")


def english():
    """The English two-voice mode: SepFormer, both Whisper small.en recognizers, the active profile, the demo audio."""
    paths = []
    for folder in ("candidates-v3", "whisper-small.en-cleartalk"):
        paths += sorted(p for p in (ROOT / "models" / folder).rglob("*") if p.is_file())
    paths += [p for p in (ROOT / "models" / "active_profile.json",) if p.is_file()]
    paths += sorted(p for p in (ROOT / "data" / "demo").glob("*.wav") if p.name != "conversation.wav")
    if not (ROOT / "models" / "whisper-small.en-cleartalk").is_dir():
        print("note: models/whisper-small.en-cleartalk is not on this computer; the archive has the pretrained English models only")
    out = ROOT / "dist" / "ClearTalk_english_models.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"{out} · {len(paths)} files · {out.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
