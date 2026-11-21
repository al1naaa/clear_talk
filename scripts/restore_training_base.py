"""Rebuild a trainable merged model and verify its CT2 binary against deployed step 600.

The original optimizer is unavailable. A fresh LoRA is fitted on this verified
initialization. Run on the trusted local transfer with CT2 FP16 format v6.
"""

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from ctranslate2.converters.transformers import WhisperLoader
from transformers import WhisperForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def decoder_weights(path):
    with path.open("rb") as stream:

        def num(fmt):
            return struct.unpack("<" + fmt, stream.read(struct.calcsize("<" + fmt)))

        def name():
            return stream.read(num("H")[0])[:-1].decode()

        if num("I")[0] != 6 or name() != "WhisperSpec" or num("I")[0] != 3:
            raise ValueError("Expected CT2 Whisper v6, spec revision 3")
        for _ in range(num("I")[0]):
            key = name()
            shape = num("I" * num("B")[0])
            dtype, size = num("B")[0], num("I")[0]
            if (
                key.startswith("decoder/layer_")
                and key.endswith("/weight")
                and "/linear_" in key
            ):
                assert dtype == 4 and size == np.prod(shape) * 2
                yield (
                    key,
                    torch.from_numpy(
                        np.frombuffer(stream.read(size), dtype="<f2")
                        .reshape(shape)
                        .copy()
                    ),
                )
            else:
                stream.seek(size, 1)


@torch.no_grad()
def restore(base, deployed, output):
    model = WhisperForConditionalGeneration.from_pretrained(base, dtype=torch.float32)
    pieces = {
        ("attention", "linear_0"): ("encoder_attn.q_proj",),
        ("attention", "linear_1"): ("encoder_attn.k_proj", "encoder_attn.v_proj"),
        ("attention", "linear_2"): ("encoder_attn.out_proj",),
        ("self_attention", "linear_0"): (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ),
        ("self_attention", "linear_1"): ("self_attn.out_proj",),
        ("ffn", "linear_0"): ("fc1",),
        ("ffn", "linear_1"): ("fc2",),
    }
    seen = set()
    for key, weight in decoder_weights(deployed / "model.bin"):
        _, layer, block, proj, _ = key.split("/")
        names = pieces[block, proj]
        layer_id = int(layer.removeprefix("layer_"))
        for dest, value in zip(names, weight.chunk(len(names)), strict=True):
            target = model.model.decoder.layers[layer_id].get_submodule(dest).weight
            assert target.shape == value.shape
            target.copy_(value)
        seen.add(key)
    assert len(seen) == 7 * model.config.decoder_layers
    output.mkdir(parents=True, exist_ok=True)
    # Check every retained tensor, alias and dimension by converting back through CT2.
    spec = WhisperLoader().get_model_spec(model)
    spec.register_vocabulary(json.loads((deployed / "vocabulary.json").read_text()))
    spec.validate()
    spec.optimize(quantization="float16")
    check = output / "roundtrip.bin"
    spec._serialize(str(check))
    expected = digest(deployed / "model.bin")
    actual = digest(check)
    if actual != expected:
        raise ValueError(f"Whole-model FP16 round trip differs: {actual} vs {expected}")
    check.unlink()
    model.save_pretrained(output, safe_serialization=True)
    for file in ("tokenizer.json", "preprocessor_config.json"):
        shutil.copyfile(deployed / file, output / file)
    shutil.copyfile(base / "tokenizer_config.json", output / "tokenizer_config.json")
    (output / "verified_sha256.txt").write_text(actual + "\n")
    print("Verified complete initialized model", actual, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "models/training-kk-turbo")
    parser.add_argument(
        "--deployed",
        type=Path,
        default=ROOT / "models/multilingual/whisper-large-cleartalk-multilingual",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models/training-round2")
    parser.add_argument(
        "--fetch-base",
        action="store_true",
        help="Download the pinned native Kazakh starting weights",
    )
    opt = parser.parse_args()
    if opt.fetch_base:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "shyngys879/kazakh-whisper-large-v3-turbo",
            revision="dafae810c95496f66184605824be5a0a971d3c09",
            local_dir=opt.base,
            allow_patterns=["*.safetensors", "*.json"],
            token=False,
        )
    restore(opt.base, opt.deployed, opt.output)
