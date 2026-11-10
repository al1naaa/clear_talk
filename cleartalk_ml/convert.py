"""Convert a downloaded Whisper model (PyTorch) to the CTranslate2 format the app runs.

    python -m cleartalk_ml.convert turbo kazakh small

Weights are stored as float16 (half the size); the app converts them when loading: int8 on a CPU,
float32 or float16 on a graphics card. The folder works unchanged on Windows and macOS.
"""

import argparse
import hashlib
import json
import shutil

from .paths import MODELS, READY_MODELS

CT2_FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.json", "preprocessor_config.json")


def sha256(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def same_vocabulary(first, second) -> bool:
    """True when two tokenizer.json files define the same tokens with the same ids."""
    def load(path):
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["model"]["vocab"], {token["content"]: token["id"] for token in data["added_tokens"]}

    return load(first) == load(second)


def convert(source, output, card=None, force=False, tokenizer_source=None):
    """Convert ``source``; ``tokenizer_source`` supplies the tokenizer files when the model's own
    tokenizer settings were written by a newer library than the one installed (the vocabulary must
    be identical, which is checked)."""
    from ctranslate2.converters import TransformersConverter

    if (output / "model.bin").is_file() and (output / "model_card.json").is_file() and not force:
        return output
    tokenizer_source = tokenizer_source or source
    if tokenizer_source != source and not same_vocabulary(source / "tokenizer.json", tokenizer_source / "tokenizer.json"):
        raise ValueError(f"{tokenizer_source.name} does not have the same vocabulary as {source.name}")

    class Converter(TransformersConverter):
        def load_tokenizer(self, tokenizer_class, model_name_or_path, **kwargs):
            return tokenizer_class.from_pretrained(str(tokenizer_source), **kwargs)

    Converter(str(source), copy_files=["preprocessor_config.json"], low_cpu_mem_usage=True).convert(
        str(output), quantization="float16", force=True)
    shutil.copy(tokenizer_source / "tokenizer.json", output / "tokenizer.json")
    card = dict(card or {})
    if tokenizer_source != source:
        card["tokenizer_from"] = tokenizer_source.name
    card["files_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir())
                            if p.is_file() and p.name != "model_card.json"}
    (output / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", choices=sorted(READY_MODELS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for key in args.models:
        entry = READY_MODELS[key]
        print(f"converting {entry['name']} ...", flush=True)
        print(convert(MODELS / entry["hf"], MODELS / entry["ct2"],
                      {"name": entry["name"], "source": entry["hf"], "kind": "ready model, not trained here"},
                      args.force, MODELS / entry["tokenizer"] if entry.get("tokenizer") else None))


if __name__ == "__main__":
    main()
