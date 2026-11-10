"""Shared, fixed ASR policy for serving and v3 development evaluation.

The VAD is a speech-activity detector, not a guarantee that a clip contains no
speech. In particular, very quiet or heavily distorted speech can be missed.
No recognizer output is fabricated when the detector finds no speech.
"""

from io import BytesIO
from pathlib import Path

import numpy as np

from .loops import cut_loops


SAMPLE_RATE = 16000
VAD_PARAMETERS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 160,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 200,
}
POLICY_ID = "english-vad-v1"


def decoding_policy() -> dict:
    """Return a fresh dictionary so callers cannot mutate the shared policy."""
    return {
        "language": "en",
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": dict(VAD_PARAMETERS),
        "condition_on_previous_text": False,
        "temperature": 0.0,
    }


def source_to_asr_audio(source, sample_rate: int = 8000) -> np.ndarray:
    """Exactly mirror serving's source peak-normalization / PCM16 / decoding.

    Serving saves a mono source with SoundFile at its native 8 kHz before the
    recognizer decodes/resamples it. Evaluation uses this in-memory equivalent,
    avoiding different resamplers or a slightly different PCM16 quantizer.
    """
    import soundfile as sf
    from faster_whisper.audio import decode_audio

    source = np.asarray(source, dtype=np.float32).copy()
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("A source must be a non-empty finite mono waveform")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    peak = float(np.max(np.abs(source)))
    if peak > 1e-6:
        source *= 0.95 / peak
    stream = BytesIO()
    sf.write(stream, source, sample_rate, format="WAV", subtype="PCM_16")
    stream.seek(0)
    return decode_audio(stream, sampling_rate=SAMPLE_RATE)


def _no_speech_result(reason: str) -> dict:
    return {
        "text": "",
        "segments": [],
        "review_notes": [reason],
        "no_speech": True,
        "speech_detected": False,
        "speech_duration_seconds": 0.0,
        "transcription_status": "no_speech_detected",
        "asr_policy": POLICY_ID,
    }


def transcribe_audio(model, audio, duration: float | None = None) -> dict:
    """Transcribe mono float audio at 16 kHz with a no-speech abstention path.

    ``model`` is an already-loaded faster-whisper WhisperModel. Detecting no
    speech returns without iterating the lazy recognizer segment generator.
    Suspect ASR scores flag text for review; they do not arbitrarily delete it.
    The unchanged v2 benchmark intentionally does NOT call this policy.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("ASR input must be a finite mono waveform at 16 kHz")
    actual_duration = len(audio) / SAMPLE_RATE
    if duration is None:
        duration = actual_duration
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and non-negative")
    duration = min(float(duration), actual_duration)
    if audio.size == 0 or float(np.max(np.abs(audio))) <= 1e-8:
        return _no_speech_result("The recording is empty or digitally silent; no words were transcribed.")

    segments, info = model.transcribe(audio, **decoding_policy())
    speech_duration = float(info.duration_after_vad)
    if not np.isfinite(speech_duration) or speech_duration < 0:
        raise RuntimeError("The speech detector returned invalid duration metadata")
    if speech_duration == 0:
        return _no_speech_result(
            "No speech was detected. Quiet or heavily distorted speech can be missed; "
            "listen to the recording before concluding that it contains no speech."
        )

    text_parts, items, review_notes = [], [], []
    for segment in segments:
        text, looped = cut_loops(segment.text.strip())
        if not text:
            continue
        start = min(duration, max(0.0, float(segment.start)))
        end = min(duration, max(start, float(segment.end)))
        needs_review = segment.avg_logprob < -0.9 or segment.no_speech_prob > 0.6 or looped
        items.append({"start": round(start, 2), "end": round(end, 2), "text": text, "needs_review": needs_review})
        text_parts.append(text)
        if needs_review:
            review_notes.append(
                f"Check {start:.1f}-{end:.1f}s: an ASR score crossed a review threshold. "
                "This is a heuristic, not a calibrated confidence estimate."
            )
    if not items:
        review_notes.append("Speech activity was detected, but no words were transcribed. Listen to the recording to review it.")
    return {
        "text": " ".join(text_parts),
        "segments": items,
        "review_notes": review_notes,
        "no_speech": False,
        "speech_detected": True,
        "speech_duration_seconds": round(min(speech_duration, actual_duration), 3),
        "transcription_status": "transcribed" if items else "no_transcript",
        "asr_policy": POLICY_ID,
    }


def transcribe_path(model, path: Path | str, duration: float | None = None) -> dict:
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    return transcribe_audio(model, audio, duration)
