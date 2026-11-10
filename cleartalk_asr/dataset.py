"""Whisper inputs and targets for separated tracks.

Audio is decoded with faster-whisper's decoder, exactly as the app decodes a saved
track, then turned into Whisper's fixed 80 x 3000 log-mel features. Targets follow
Whisper's own format; half of the training tracks keep segment timestamps so the
adapted model still produces the timed segments the app displays.
"""

import hashlib
from functools import lru_cache

import numpy as np
import torch

from .paths import BASE_WHISPER, ROOT

SAMPLE_RATE = 16000
MAX_LABEL_TOKENS = 440  # Whisper's decoder holds 448 positions


@lru_cache(maxsize=1)
def processor():
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(BASE_WHISPER)


def load_audio(relative_path):
    from faster_whisper.audio import decode_audio

    return decode_audio(str(ROOT / relative_path), sampling_rate=SAMPLE_RATE)


def features(audio):
    return processor().feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]


def uses_timestamps(track_id, fraction=0.5):
    """Fixed, reproducible half of the tracks (by id hash) are timestamped targets."""
    digest = int(hashlib.sha256(track_id.encode()).hexdigest()[:8], 16)
    return digest / 0xFFFFFFFF < fraction


def target_tokens(text, span=None):
    """[<|startoftranscript|>, <|notimestamps|> | <|t0|>, text..., (<|t1|>), <|endoftext|>]."""
    tokenizer = processor().tokenizer
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    text_ids = tokenizer.encode(" " + text.strip(), add_special_tokens=False)
    if span is None:
        tokens = [sot, tokenizer.convert_tokens_to_ids("<|notimestamps|>"), *text_ids, eot]
    else:
        begin = tokenizer.convert_tokens_to_ids("<|0.00|>")
        start, end = (min(max(round(float(t) / 0.02), 0), 1500) for t in span)
        end = max(end, start + 1)
        tokens = [sot, begin + start, *text_ids, begin + min(end, 1500), eot]
    return tokens[:MAX_LABEL_TOKENS]


class TrackDataset(torch.utils.data.Dataset):
    def __init__(self, frame, timestamp_fraction=0.5):
        self.rows = frame.to_dict("records")
        self.timestamp_fraction = timestamp_fraction

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        span = row["span"] if uses_timestamps(row["track_id"], self.timestamp_fraction) else None
        return {"input_features": features(load_audio(row["path"])).astype(np.float32),
                "tokens": target_tokens(row["text"], span)}


def collate(batch):
    eot = processor().tokenizer.convert_tokens_to_ids("<|endoftext|>")
    longest = max(len(item["tokens"]) for item in batch) - 1
    decoder_inputs = torch.full((len(batch), longest), eot, dtype=torch.long)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        tokens = torch.tensor(item["tokens"], dtype=torch.long)
        decoder_inputs[row, :len(tokens) - 1] = tokens[:-1]
        labels[row, :len(tokens) - 1] = tokens[1:]
    return {"input_features": torch.from_numpy(np.stack([item["input_features"] for item in batch])),
            "decoder_input_ids": decoder_inputs, "labels": labels}
