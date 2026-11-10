"""Decode only the required audio stream, with an early duration limit."""

from pathlib import Path


class AudioValidationError(ValueError):
    pass


def decode_limited(path: Path, sample_rate: int, max_seconds: float):
    """Return mono float32 samples without decoding an arbitrarily long upload.

    PyAV bundles its decoding libraries, so a separate FFmpeg executable is not
    needed. Video is ignored: video containers are supported as audio inputs.
    """
    import av
    import numpy as np

    parts = []
    count = 0
    limit = int(max_seconds * sample_rate)
    too_long = (f"Please trim the recording to {max_seconds:g} seconds or less." if max_seconds <= 120
                else f"Please trim the recording to {max_seconds / 60:g} minutes or less.")
    try:
        with av.open(str(path)) as container:
            streams = list(container.streams.audio)
            if not streams:
                raise AudioValidationError("This file does not contain an audio track.")
            stream = streams[0]
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
            for frame in container.decode(stream):
                frame.pts = None
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().reshape(-1)
                    count += len(samples)
                    if count > limit:
                        raise AudioValidationError(too_long)
                    parts.append(samples)
            for converted in resampler.resample(None):
                samples = converted.to_ndarray().reshape(-1)
                count += len(samples)
                if count > limit:
                    raise AudioValidationError(too_long)
                parts.append(samples)
    except AudioValidationError:
        raise
    except Exception as exc:
        raise AudioValidationError("This audio could not be decoded. Try a WAV, MP3, M4A or FLAC file.") from exc

    if not parts or count < int(0.25 * sample_rate):
        raise AudioValidationError("Please provide at least a quarter-second of audio.")
    audio = np.concatenate(parts).astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        raise AudioValidationError("This file contains invalid audio samples.")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-5:
        raise AudioValidationError("This recording is silent. Please choose a recording with speech.")
    return np.clip(audio, -1.0, 1.0)
