"""Short quiet stretches that are background voices or noise, not the conversation (Conversation mode).

In noisy recordings, background talkers between and around the real turns pass speech detection, get a speaker label
of their own, and the recogniser writes something for them: a few invented words per stretch ("episode one of the
first season"). On the validation recordings with background talkers these stretches were short (median 1.5 s),
clearly quieter than the conversation, and barely above the recording's background level, while real turns, even
quiet ones, stand well above it. A turn is set aside as background when all three hold.

The thresholds were chosen on one half of the robustness validation recordings and confirmed on the other half
(see ml_pipeline/round4/PIPELINE.md).
"""

import numpy as np

FRAME = 800                 # 50 ms at 16 kHz
BELOW_SPEECH_DB = -5.0      # the turn's loud frames are at least 5 dB below the recording's speech level ...
MAX_SECONDS = 3.0           # ... it is at most 3 s long ...
ABOVE_FLOOR_DB = 10.0       # ... and at most 10 dB above the recording's background level


class Levels:
    """Frame levels of a whole recording, measured once."""

    def __init__(self, audio, sample_rate=16000):
        count = len(audio) // FRAME
        self.frames_per_second = sample_rate / FRAME
        if count == 0:
            self.db = np.zeros(0)
            self.speech = self.floor = 0.0
            return
        energy = np.sqrt(np.mean(np.asarray(audio[: count * FRAME], dtype=np.float32).reshape(count, FRAME) ** 2, axis=1))
        self.db = 20 * np.log10(energy + 1e-8)
        self.speech = float(np.percentile(self.db, 95))
        self.floor = float(np.percentile(self.db, 20))

    def is_background(self, start, end) -> bool:
        if end - start > MAX_SECONDS or not len(self.db):
            return False
        a = int(start * self.frames_per_second)
        frames = self.db[a: max(int(end * self.frames_per_second), a + 1)]
        if not len(frames):
            return False
        top = float(np.percentile(frames, 90))
        return top - self.speech <= BELOW_SPEECH_DB and top - self.floor <= ABOVE_FLOOR_DB
