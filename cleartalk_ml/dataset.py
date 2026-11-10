"""Whisper inputs and targets for single-language utterances and mixed-language recordings.

Features (step 7): Whisper's own fixed log-mel spectrogram, 3000 frames (30 s) of 80 bands for the small
model and 128 for the large one. Nothing is fitted, so nothing can leak from validation or test. Audio is
decoded exactly as the app decodes it.

Targets (step 8): Whisper's own token format,

    <|startoftranscript|> <|language|> <|transcribe|> [<|notimestamps|>] text <|endoftext|>

* A single-language utterance carries its own language token.
* A mixed-language recording carries ONE language token although several languages follow - the app
  detects one language per window, so the model has to learn to keep writing each part in its own
  language whatever the token says. The token is the first part's language in 70% of the recordings
  and another language present in the recording otherwise (fixed by a hash of the recording id).
* Half of the examples (fixed by a hash of the id) keep timestamps: one timed segment per utterance,
  so in a mixed recording the segment boundaries fall on the language changes.
"""

import hashlib
from functools import lru_cache

import numpy as np

from .paths import BASES, MODELS, SAMPLE_RATE, base_name

MAX_LABEL_TOKENS = 440       # Whisper's decoder holds 448 positions
TIMESTAMP_FRACTION = 0.5
FIRST_LANGUAGE_PROMPT = 0.7  # share of mixed recordings prompted with their first language


@lru_cache(maxsize=4)
def _processor(base):
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(MODELS / BASES[base]["processor"])


def processor():
    """Feature extractor and tokenizer of the current base model (80 mel bands for small, 128 for large)."""
    return _processor(base_name())


def features(audio):
    return processor().feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]


def _unit(key: str) -> float:
    """A reproducible number in [0, 1) from an id (no random state involved)."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0x100000000


def uses_timestamps(example_id: str, fraction: float = TIMESTAMP_FRACTION) -> bool:
    return _unit("timestamps:" + example_id) < fraction


def prompt_language(example) -> str:
    """The language token of an example (see the module text for mixed recordings)."""
    parts = example.get("parts")
    if not parts:
        return example["language"]
    first = parts[0]["language"]
    others = sorted({part["language"] for part in parts} - {first})
    if not others or _unit("prompt:" + example["id"]) < FIRST_LANGUAGE_PROMPT:
        return first
    return others[int(_unit("other:" + example["id"]) * len(others))]


def target_tokens(example, timestamps=None) -> list[int]:
    """Decoder tokens of one example; ``example`` has id, language/text (single) or parts (mixed)."""
    tokenizer = processor().tokenizer
    token = tokenizer.convert_tokens_to_ids
    if timestamps is None:
        timestamps = uses_timestamps(example["id"])
    parts = example.get("parts") or [{"start": 0.0, "end": example["duration"], "text": example["text"]}]
    tokens = [token("<|startoftranscript|>"), token(f"<|{prompt_language(example)}|>"), token("<|transcribe|>")]
    if not timestamps:
        tokens.append(token("<|notimestamps|>"))
        text = " ".join(part["text"].strip() for part in parts)
        tokens.extend(tokenizer.encode(" " + text, add_special_tokens=False))
    else:
        zero = token("<|0.00|>")
        for part in parts:
            start = min(max(round(float(part["start"]) / 0.02), 0), 1500)
            end = min(max(round(float(part["end"]) / 0.02), start + 1), 1500)
            tokens.append(zero + start)
            tokens.extend(tokenizer.encode(" " + part["text"].strip(), add_special_tokens=False))
            tokens.append(zero + end)
    tokens = tokens[:MAX_LABEL_TOKENS - 1]
    return [*tokens, token("<|endoftext|>")]


def example_audio(example) -> np.ndarray:
    """Mono 16 kHz audio of an example; mixed recordings are rendered from their recipe on the fly."""
    from .corpora import read_audio
    from .mixed import render

    if example.get("parts"):
        audio = render({"duration": example["duration"], "parts": example["parts"]})
    else:
        audio = read_audio(example["path"])
    if example.get("noise"):  # a noisy copy for a monitor: the example carries its noise and its talkers
        from .noise import add

        audio = add(audio, example["noise"], example["noise"].get("talkers", []))
    return audio


class ExampleDataset:
    """Indexable examples -> {"input_features", "tokens"}; works as a torch map-style dataset."""

    def __init__(self, examples, timestamp_fraction=TIMESTAMP_FRACTION, noise_probability=0.0, talkers=()):
        self.examples = list(examples)
        self.timestamp_fraction = timestamp_fraction
        self.noise_probability = noise_probability   # round 4: this share of the examples gets background noise
        self.talkers = list(talkers)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        stamps = uses_timestamps(example["id"], self.timestamp_fraction)
        audio = example_audio(example)
        if self.noise_probability > 0:
            from .noise import add, spec_for

            audio = add(audio, spec_for(example["id"], self.noise_probability), self.talkers)
        return {"input_features": features(audio).astype(np.float32), "tokens": target_tokens(example, stamps)}


def collate(batch):
    import torch

    eot = processor().tokenizer.convert_tokens_to_ids("<|endoftext|>")
    longest = max(len(item["tokens"]) for item in batch) - 1
    decoder_inputs = torch.full((len(batch), longest), eot, dtype=torch.long)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        tokens = torch.tensor(item["tokens"], dtype=torch.long)
        decoder_inputs[row, :len(tokens) - 1] = tokens[:-1]
        labels[row, :len(tokens) - 1] = tokens[1:]
        labels[row, :2] = -100  # the language and task tokens are given at inference, not predicted
    return {"input_features": torch.from_numpy(np.stack([item["input_features"] for item in batch])),
            "decoder_input_ids": decoder_inputs, "labels": labels}
