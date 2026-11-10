"""Merge a LoRA adapter into Whisper small.en and convert it for the app (faster-whisper).

    python -m cleartalk_asr.export --adapter models/asr-runs/<run>/best

The merged model has exactly the base architecture, so it runs in the app at the
same speed as the pretrained recognizer. Weights are stored in float16 like the
pretrained faster-whisper package; the app quantizes both to int8 when loading.
"""

import argparse
import hashlib
import json
import shutil

from .paths import BASE_WHISPER, EXPORT, ROOT


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export(adapter_dir, output_dir=EXPORT, card=None):
    import torch
    from ctranslate2.converters import TransformersConverter
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    adapter_dir = ROOT / adapter_dir
    merged_dir = adapter_dir.parent / f"merged-{adapter_dir.name}"
    base = WhisperForConditionalGeneration.from_pretrained(BASE_WHISPER, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    WhisperProcessor.from_pretrained(BASE_WHISPER).save_pretrained(merged_dir)
    shutil.copy(BASE_WHISPER / "generation_config.json", merged_dir / "generation_config.json")
    TransformersConverter(str(merged_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
        str(output_dir), quantization="float16", force=True)
    card = dict(card or {})
    card.update(adapter=str(adapter_dir.relative_to(ROOT)), base_model="openai/whisper-small.en",
                files_sha256={p.name: sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "model_card.json"})
    (output_dir / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default=str(EXPORT))
    parser.add_argument("--card", help="JSON file with training/validation facts for the model card")
    args = parser.parse_args()
    card = json.loads((ROOT / args.card).read_text()) if args.card else None
    print(export(args.adapter, ROOT / args.output, card))


if __name__ == "__main__":
    main()
