"""Two people talking at once (Conversation mode): find the overlap and split the stretch into two voices.

The two-voice separator of the English mode (SepFormer, 8 kHz) is language-independent enough to untangle Kazakh,
Russian and English voices. A stretch of speech is separated only when the speaker labels show two people in it
(each talking for at least half a second); it counts as two voices when the quieter separated track still reaches
within 12 dB of the louder one at its loud moments (a faint copy of one voice does not). Each track is then
recognised on its own and given to one of the two speakers, and a sentence is kept only from the track that is
louder at that moment.

Thresholds set on calibration recordings (overlap recordings 0-49 and mixed validation recordings 100-149, see the
round-4 log): the speaker labels showed two people in 30 of the 48 stretches with overlap and in 2 of the 226
without; all 30 had the quieter track within 9.4 dB, one of the two others at -16.7 dB.
"""

import numpy as np

SEPARATOR_RATE = 8000
FRAME = 400                    # 50 ms at 8 kHz
MIN_SPEAKER_SECONDS = 0.5      # a second speaker label counts in a stretch when it talks this long there
MIN_PEAK_RATIO_DB = -12.0      # the quieter track's loud frames within this of the louder track's
MAX_SECONDS = 30.0             # longer stretches are not separated: the separator's memory grows with length, and a
                               # minute of "two people" is background chatter rather than two people interrupting
LEAK_DB = -10.0                # a track's sentence is kept when the track is not this much quieter than the other there


def peak_ratio(track_a, track_b) -> float:
    """Level of the quieter track's loud frames (95th percentile of 50 ms frames) against the louder track's, in dB."""
    count = min(len(track_a), len(track_b)) // FRAME
    if count == 0:
        return -99.0
    peaks = sorted(float(np.percentile(20 * np.log10(np.sqrt(np.mean(t[: count * FRAME].reshape(count, FRAME) ** 2, axis=1)) + 1e-8), 95))
                   for t in (track_a, track_b))
    return peaks[0] - peaks[1]


def is_overlap(ratio_db) -> bool:
    return ratio_db >= MIN_PEAK_RATIO_DB


SAME_WORDS = 0.5               # two tracks whose texts share this much of the shorter one carry one voice twice
# Round 5, the speaker check (set on calibration recordings, see ml_pipeline/round5/PIPELINE.md): each separated track
# must sound like a different one of the two people the speaker labels give in the stretch.
MIN_VOICE_MATCH = 0.5          # voice-print similarity of each track with the person it is given to
MIN_VOICE_MARGIN = 0.3         # ... and how much closer it is to that person than to the other one


def same_words(text_a: str, text_b: str) -> bool:
    """Whether the two separated tracks were written as largely the same words: then the separator spread one
    voice over both tracks (two people talking at once do not say the same words), and the stretch is not separated."""
    import re
    from difflib import SequenceMatcher

    words = [[w for w in (re.sub(r"[^\w]", "", w.lower()) for w in text.split()) if w] for text in (text_a, text_b)]
    shorter = min(len(words[0]), len(words[1]))
    if shorter == 0:
        return False
    matched = sum(block.size for block in SequenceMatcher(None, words[0], words[1], autojunk=False).get_matching_blocks())
    return matched / shorter >= SAME_WORDS


def band_limited(audio_16k):
    """The audio as the separator's 8 kHz tracks sound (so voice prints of tracks and of the mixture compare)."""
    from scipy.signal import resample_poly

    return resample_poly(resample_poly(np.asarray(audio_16k, dtype=np.float32), 1, 2), 2, 1).astype(np.float32)[: len(audio_16k)]


def active_spans(track, rate=16000, below_db=-20.0, shortest=0.4):
    """(start, end) seconds where a track carries its own voice: 50 ms frames within ``below_db`` of its loud frames."""
    frame = rate // 20
    count = len(track) // frame
    if count == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(np.asarray(track[: count * frame]).reshape(count, frame) ** 2, axis=1)) + 1e-8)
    active = db > np.percentile(db, 95) + below_db
    spans, start = [], None
    for index, flag in enumerate(np.append(active, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if spans and start - spans[-1][1] <= 4:      # close gaps up to 0.2 s
                spans[-1] = (spans[-1][0], index)
            else:
                spans.append((start, index))
            start = None
    return [(a / 20, b / 20) for a, b in spans if (b - a) / 20 >= shortest]


def voice_print(encoder, audio, spans, window=1.5, hop=0.75):
    """Mean unit voice embedding over the given stretches (cut into windows), or None when there is too little audio."""
    windows = []
    for start, end in spans:
        if end - start < 0.4:
            continue
        cursor = start
        while cursor + window < end:
            windows.append((cursor, cursor + window))
            cursor += hop
        windows.append((max(start, end - window), end))
    if not windows:
        return None
    vector = encoder.embed(audio, windows).mean(axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def who_is_who(track_prints, speaker_prints):
    """Which of the two speakers each separated track is ({track: index into speaker_prints}), or None when the tracks do
    not sound like two different people of the conversation (one voice spread over both tracks, or background talkers)."""
    if any(v is None for v in list(track_prints) + list(speaker_prints)):
        return None
    sims = [[float(np.dot(t, s)) for s in speaker_prints] for t in track_prints]
    straight, crossed = sims[0][0] + sims[1][1], sims[0][1] + sims[1][0]
    mapping = {0: 0, 1: 1} if straight >= crossed else {0: 1, 1: 0}
    for track, speaker in mapping.items():
        if sims[track][speaker] < MIN_VOICE_MATCH or sims[track][speaker] - sims[track][1 - speaker] < MIN_VOICE_MARGIN:
            return None
    return mapping


class Separator:
    """SepFormer from the English mode's model folder, loaded on first use."""

    def __init__(self, settings, device="cpu"):
        self.settings, self.device, self.model = settings, device, None

    def split(self, audio_16k):
        """The two 16 kHz tracks of a mono 16 kHz stretch (as separated, and levelled for the recogniser), whether
        they are two voices at once, and the measurements behind that decision."""
        import torch
        from scipy.signal import resample_poly

        audio_16k = np.asarray(audio_16k, dtype=np.float32)
        if audio_16k.ndim != 1 or not audio_16k.size or not np.isfinite(audio_16k).all():
            raise ValueError("Separation input must be a nonempty finite mono waveform at 16 kHz")
        if len(audio_16k) > int(MAX_SECONDS * 16000):
            raise ValueError(f"Two-voice separation supports at most {MAX_SECONDS:g} seconds")
        if self.model is None:
            self.settings.verify_separator()
            from speechbrain.inference.separation import SepformerSeparation

            self.model = SepformerSeparation.from_hparams(source=str(self.settings.separator_dir), savedir=str(self.settings.separator_dir),
                                                          run_opts={"device": self.device})
            self.model.mods.eval()
        mixture = torch.from_numpy(resample_poly(audio_16k, SEPARATOR_RATE, 16000).astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model.separate_batch(mixture).detach().cpu().numpy()
        expected_shape = (1, mixture.shape[1], 2)
        if output.shape != expected_shape or not np.isfinite(output).all():
            raise RuntimeError("The separator did not return two finite tracks matching the input duration")
        tracks = output[0]
        del mixture
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()   # hand the memory back for the recogniser, which has its own allocator
        ratio = peak_ratio(tracks[:, 0], tracks[:, 1])
        raw = [resample_poly(tracks[:, k], 16000, SEPARATOR_RATE).astype(np.float32)[: len(audio_16k)] for k in range(2)]
        if any(track.shape != audio_16k.shape or not np.isfinite(track).all() for track in raw):
            raise RuntimeError("Resampling did not preserve the two finite source tracks")
        loud = [track * (0.5 / (float(np.max(np.abs(track))) or 1.0)) for track in raw]   # for the recogniser
        return raw, loud, is_overlap(ratio), {"peak_ratio_db": round(ratio, 1)}

    @staticmethod
    def own_voice(tracks, track, start, end) -> bool:
        """Whether ``track`` carries its own voice in [start, end] s rather than leakage of the other track."""
        if track not in (0, 1) or len(tracks) != 2 or not np.isfinite([start, end]).all():
            return False
        limit = min(len(tracks[0]), len(tracks[1]))
        a, b = min(limit, max(0, int(start * 16000))), min(limit, max(0, int(end * 16000)))
        if b <= a:
            return False
        mine, other = (float(np.mean(np.asarray(tracks[k][a:b], dtype=np.float64) ** 2)) for k in (track, 1 - track))
        if not np.isfinite([mine, other]).all():
            return False
        if mine <= 0.0:
            return False
        return other <= 0.0 or 10 * np.log10(mine / other) >= LEAK_DB

    def dominant(self, tracks, spans):
        """Which track is louder during the given (start, end) spans in seconds; None when the spans hold no audio."""
        scores = [0.0, 0.0]
        for start, end in spans:
            a, b = int(start * 16000), int(end * 16000)
            for k in range(2):
                scores[k] += float(np.sum(tracks[k][a:b] ** 2))
        if scores[0] == scores[1] == 0.0:
            return None
        return 0 if scores[0] >= scores[1] else 1
