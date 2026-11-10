"""Merge a LoRA adapter into multilingual Whisper small and convert it for the app (faster-whisper).

    python -m cleartalk_ml.export --adapter models/multilingual/runs/<run>/step-00900

The merged model has exactly the base architecture, so it runs in the app at the speed of Whisper
small. Weights are stored as float16; the app converts them when loading (int8 on a CPU). The output
folder is plain files and works unchanged on Windows and macOS.
"""

import argparse
import json
import shutil
from pathlib import Path

from .convert import sha256
from .paths import BASES, MODELS, ROOT

EXPORT = MODELS / "whisper-small-cleartalk-multilingual"


def export(adapter_dir, output_dir=EXPORT, card=None, keep_merged=False, base="small"):
    import torch
    from ctranslate2.converters import TransformersConverter
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    adapter_dir, output_dir = Path(adapter_dir), Path(output_dir)
    merged_dir = adapter_dir.parent / f"merged-{adapter_dir.name}"
    source, processor_dir = MODELS / BASES[base]["hf"], MODELS / BASES[base]["processor"]
    weights = WhisperForConditionalGeneration.from_pretrained(source, dtype=torch.float32)
    model = PeftModel.from_pretrained(weights, adapter_dir).merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    WhisperProcessor.from_pretrained(processor_dir).save_pretrained(merged_dir)
    shutil.copy(processor_dir / "generation_config.json", merged_dir / "generation_config.json")
    TransformersConverter(str(merged_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
        str(output_dir), quantization="float16", force=True)
    if not keep_merged:
        shutil.rmtree(merged_dir)  # an intermediate copy; the adapter and the base model rebuild it
    card = dict(card or {})
    try:
        card["adapter"] = adapter_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        card["adapter"] = adapter_dir.name
    card.update(base_model=BASES[base]["name"],
                files_sha256={p.name: sha256(p) for p in sorted(output_dir.iterdir())
                              if p.is_file() and p.name != "model_card.json"})
    (output_dir / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default=str(EXPORT))
    parser.add_argument("--card", help="JSON file with training/validation facts for the model card")
    parser.add_argument("--base", choices=sorted(BASES), default="small")
    args = parser.parse_args()
    card = json.loads((ROOT / args.card).read_text(encoding="utf-8")) if args.card else None
    print(export(ROOT / args.adapter, ROOT / args.output, card, base=args.base))


if __name__ == "__main__":
    main()
