"""Who spoke when, for any number of speakers taking turns - built from ungated parts.

1. Speech detection (Silero VAD, bundled with faster-whisper) finds the stretches with speech.
2. Each stretch is cut into short overlapping windows; a speaker-embedding model
   (ECAPA-TDNN trained on VoxCeleb, from SpeechBrain) turns every window into a voice "fingerprint".
3. Windows with similar fingerprints are grouped (agglomerative clustering, cosine distance);
   each group is one speaker. The number of speakers is found automatically unless it is given.
4. Neighbouring windows of one speaker become a turn.

Limits: speakers talking at the same time are not separated (a window gets one label), very short
interjections can be attached to the wrong person, and "Speaker 1 / 2 / …" are labels inside one
recording, not identified people.
"""

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
VAD_OPTIONS = {"threshold": 0.5, "min_speech_duration_ms": 200, "min_silence_duration_ms": 300,
               "speech_pad_ms": 100, "max_speech_duration_s": 20.0}
WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.75
MIN_WINDOW_SECONDS = 0.4
# Cosine distance at which two groups of windows stop being merged; chosen on validation
# conversations (see ml_pipeline/PIPELINE.md, "Speaker labels").
DISTANCE_THRESHOLD = 0.70
MAX_SPEAKERS = 8
MIN_SPEAKER_SHARE = 0.02   # a "speaker" with less than this share of the windows is absorbed
MERGE_GAP_SECONDS = 0.6
MAX_TURN_SECONDS = 24.0
LINKAGE_LIMIT = 4000       # windows clustered directly; longer recordings assign the rest to centroids


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def speech_regions(audio) -> list[tuple[float, float]]:
    """Stretches of speech in seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    stamps = get_speech_timestamps(np.asarray(audio, dtype=np.float32), VadOptions(**VAD_OPTIONS))
    return [(item["start"] / SAMPLE_RATE, item["end"] / SAMPLE_RATE) for item in stamps if item["end"] > item["start"]]


def joined_regions(regions, gap=MERGE_GAP_SECONDS, longest=MAX_TURN_SECONDS, same=None) -> list[tuple[float, float]]:
    """Stretches of speech with pauses of at most ``gap`` seconds closed (a breath does not cut a sentence), each at
    most ``longest`` seconds unless a single stretch is longer by itself; with ``same(a, b)``, a pause is closed only
    when it is true for the stretches on both sides (the same person talks on both sides)."""
    out = []
    for start, end in regions:
        if (out and start - out[-1][1] <= gap and end - out[-1][0] <= longest
                and (same is None or same(out[-1], (start, end)))):
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def windows_of(regions) -> list[tuple[int, float, float]]:
    """(region index, start, end) of every embedding window."""
    out = []
    for index, (start, end) in enumerate(regions):
        if end - start <= WINDOW_SECONDS:
            if end - start >= MIN_WINDOW_SECONDS:
                out.append((index, start, end))
            continue
        cursor = start
        while cursor + WINDOW_SECONDS < end:
            out.append((index, cursor, cursor + WINDOW_SECONDS))
            cursor += HOP_SECONDS
        out.append((index, max(start, end - WINDOW_SECONDS), end))
    return out


class SpeakerEncoder:
    """ECAPA-TDNN voice embeddings from a local SpeechBrain folder.

    The three parts the embedding needs (filterbank features, per-utterance mean normalization, the
    ECAPA network) are built from the folder's own hyperparams.yaml and its weights are loaded
    directly. SpeechBrain's ready-made wrapper is not used: it links files (not permitted for an
    ordinary Windows user) and falls back to the network, and the app must stay local.
    """

    def __init__(self, directory, device="cpu", threads=None):
        from pathlib import Path

        import torch
        from hyperpyyaml import load_hyperpyyaml

        directory = Path(directory)
        if threads and device == "cpu":
            torch.set_num_threads(threads)
        with open(directory / "hyperparams.yaml", encoding="utf-8") as handle:
            parts = load_hyperpyyaml(handle, overrides={"pretrained_path": directory.as_posix()})
        self.device = device
        self.features = parts["compute_features"]
        self.normalize = parts["mean_var_norm"]
        self.model = parts["embedding_model"]
        self.model.load_state_dict(torch.load(directory / "embedding_model.ckpt", map_location="cpu", weights_only=True))
        self.model.eval().to(device)

    def embed(self, audio, spans, batch_size=64) -> np.ndarray:
        """Unit-length embeddings, one row per (start, end) span in seconds."""
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        rows = []
        for offset in range(0, len(spans), batch_size):
            chunk = spans[offset:offset + batch_size]
            pieces = [audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)] for start, end in chunk]
            longest = max(len(piece) for piece in pieces)
            batch = np.zeros((len(pieces), longest), dtype=np.float32)
            for row, piece in enumerate(pieces):
                batch[row, :len(piece)] = piece
            lengths = torch.tensor([len(piece) / longest for piece in pieces], device=self.device)
            with torch.inference_mode():
                features = self.normalize(self.features(torch.from_numpy(batch).to(self.device)), lengths)
                vectors = self.model(features, lengths)
            rows.append(vectors.squeeze(1).float().cpu().numpy())
        vectors = np.concatenate(rows) if rows else np.zeros((0, 192), dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def _centroids(vectors, labels):
    keys = sorted(set(labels.tolist()))
    centres = np.stack([vectors[labels == key].mean(axis=0) for key in keys])
    return keys, centres / np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-9)


def cluster(vectors, speakers=None, threshold=DISTANCE_THRESHOLD) -> np.ndarray:
    """A speaker number (0, 1, …) for every embedding; ``speakers`` fixes how many there are."""
    from scipy.cluster.hierarchy import fcluster, linkage

    count = len(vectors)
    if count == 0:
        return np.zeros(0, dtype=int)
    if count == 1 or speakers == 1:
        return np.zeros(count, dtype=int)
    step = max(1, int(np.ceil(count / LINKAGE_LIMIT)))
    subset = vectors[::step]
    tree = linkage(subset, method="average", metric="cosine")
    if speakers:
        labels = fcluster(tree, t=min(int(speakers), len(subset)), criterion="maxclust")
    else:
        labels = fcluster(tree, t=threshold, criterion="distance")
        sizes = {key: int((labels == key).sum()) for key in set(labels.tolist())}
        keep = [key for key, size in sorted(sizes.items(), key=lambda item: -item[1])
                if size >= max(2, MIN_SPEAKER_SHARE * len(subset))][:MAX_SPEAKERS]
        if not keep:
            keep = [max(sizes, key=sizes.get)]
        labels = np.where(np.isin(labels, keep), labels, -1)
    known = labels >= 0 if not speakers else np.ones(len(labels), dtype=bool)
    keys, centres = _centroids(subset[known], labels[known])
    # One refinement pass: every window (also those left out above) joins its nearest speaker.
    nearest = np.argmax(vectors @ centres.T, axis=1)
    return nearest.astype(int)


def _smooth(labels):
    """A single window between two windows of another speaker takes their label."""
    labels = list(labels)
    for index in range(1, len(labels) - 1):
        if labels[index - 1] == labels[index + 1] != labels[index]:
            labels[index] = labels[index - 1]
    return labels


def turns_from(regions, windows, labels) -> list[Turn]:
    """Merge window labels into turns; a region without windows inherits its neighbour's speaker."""
    turns = []
    by_region = {}
    for (region, start, end), label in zip(windows, labels):
        by_region.setdefault(region, []).append((start, end, int(label)))
    previous = 0
    for index, (region_start, region_end) in enumerate(regions):
        items = by_region.get(index)
        if not items:
            turns.append(Turn(region_start, region_end, previous))
            continue
        smoothed = _smooth([label for _, _, label in items])
        cursor = region_start
        for position in range(len(items)):
            last = position == len(items) - 1
            if last or smoothed[position + 1] != smoothed[position]:
                if last:
                    boundary = region_end
                else:
                    centre = (items[position][0] + items[position][1]) / 2
                    following = (items[position + 1][0] + items[position + 1][1]) / 2
                    boundary = min(region_end, max(cursor, (centre + following) / 2))
                if boundary > cursor:
                    turns.append(Turn(cursor, boundary, smoothed[position]))
                cursor = boundary
        previous = smoothed[-1]
    merged = []
    for turn in turns:
        if (merged and merged[-1].speaker == turn.speaker and turn.start - merged[-1].end <= MERGE_GAP_SECONDS
                and turn.end - merged[-1].start <= MAX_TURN_SECONDS):
            merged[-1].end = turn.end
        else:
            merged.append(Turn(turn.start, turn.end, turn.speaker))
    # Number speakers by first appearance, so "Speaker 1" is whoever talks first.
    order = {}
    for turn in merged:
        turn.speaker = order.setdefault(turn.speaker, len(order))
    return merged


def diarize(audio, encoder=None, speakers=None, threshold=DISTANCE_THRESHOLD) -> list[Turn]:
    """Speech turns with speaker numbers; without an encoder (or with one speaker) everything is speaker 0."""
    regions = speech_regions(audio)
    if not regions:
        return []
    if encoder is None or speakers == 1:
        return turns_from(regions, [], [])
    windows = windows_of(regions)
    vectors = encoder.embed(audio, [(start, end) for _, start, end in windows])
    return turns_from(regions, windows, cluster(vectors, speakers, threshold))
