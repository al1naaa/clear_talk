from types import SimpleNamespace

import numpy as np
import pytest

from backend.transcription import decoding_policy, source_to_asr_audio, transcribe_audio, transcribe_path


class FakeModel:
    def __init__(self, segments=(), speech_duration=1.0):
        self.segments = segments
        self.speech_duration = speech_duration
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return self.segments, SimpleNamespace(duration_after_vad=self.speech_duration)


def segment(text="Actual words", start=0.1, end=0.8, avg_logprob=-0.2, no_speech_prob=0.1):
    return SimpleNamespace(text=text, start=start, end=end, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


@pytest.mark.parametrize("audio", [np.zeros(16000), np.zeros(0), np.full(16000, 1e-10)])
def test_digital_silence_does_not_call_recognizer(audio):
    model = FakeModel([segment("Invented text")])
    result = transcribe_audio(model, audio)
    assert result["no_speech"] is True
    assert result["speech_detected"] is False
    assert result["text"] == "" and result["segments"] == []
    assert model.calls == []


def test_vad_empty_does_not_iterate_recognizer_generator():
    def forbidden_generator():
        raise AssertionError("Decoder must not run for a VAD-empty recording")
        yield  # pragma: no cover

    model = FakeModel(forbidden_generator(), speech_duration=0)
    result = transcribe_audio(model, np.ones(16000) * .01)
    assert result["no_speech"] and result["text"] == ""
    assert "can be missed" in result["review_notes"][0]


def test_speech_uses_fixed_vad_and_keeps_quiet_amplitude():
    audio = np.full(16000, 1e-5, dtype=np.float32)
    model = FakeModel([segment()])
    result = transcribe_audio(model, audio)
    assert result["text"] == "Actual words"
    assert result["no_speech"] is False and result["speech_detected"] is True
    assert model.calls[0][1] == decoding_policy()
    np.testing.assert_array_equal(model.calls[0][0], audio)


def test_suspicious_scores_are_flagged_not_deleted():
    model = FakeModel([segment(no_speech_prob=.9), segment("Uncertain words", avg_logprob=-1.1)])
    result = transcribe_audio(model, np.ones(16000) * .02)
    assert result["text"] == "Actual words Uncertain words"
    assert all(item["needs_review"] for item in result["segments"])
    assert len(result["review_notes"]) == 2
    assert all("not a calibrated confidence" in note for note in result["review_notes"])


def test_segment_times_are_clamped_and_empty_text_is_skipped():
    model = FakeModel([segment(" "), segment("Words", start=-.2, end=3)])
    result = transcribe_audio(model, np.ones(16000) * .02, duration=.75)
    assert result["segments"] == [{"start": 0., "end": .75, "text": "Words", "needs_review": False}]


def test_detected_speech_but_empty_asr_is_not_called_no_speech():
    result = transcribe_audio(FakeModel([]), np.ones(16000) * .02)
    assert result["speech_detected"] is True
    assert result["no_speech"] is False
    assert result["transcription_status"] == "no_transcript"
    assert result["text"] == ""


@pytest.mark.parametrize("audio", [np.array([np.nan]), np.array([np.inf]), np.ones((2, 3))])
def test_invalid_waveforms_rejected(audio):
    with pytest.raises(ValueError):
        transcribe_audio(FakeModel(), audio)


@pytest.mark.parametrize("duration", [-1, float("inf"), float("nan")])
def test_invalid_durations_rejected(duration):
    with pytest.raises(ValueError):
        transcribe_audio(FakeModel(), np.ones(16000), duration)


def test_decoding_policy_returns_fresh_nested_options():
    policy = decoding_policy()
    policy["vad_parameters"]["threshold"] = 1
    assert decoding_policy()["vad_parameters"]["threshold"] == .5


def test_evaluation_source_conversion_matches_serving_file(tmp_path):
    import soundfile as sf
    from faster_whisper.audio import decode_audio

    rng = np.random.default_rng(21)
    source = (rng.normal(size=24001) * .015).astype(np.float32)
    original = source.copy()
    path = tmp_path / "source.wav"
    serving_source = source * (.95 / float(np.max(np.abs(source))))
    sf.write(path, serving_source, 8000, subtype="PCM_16")
    actual = source_to_asr_audio(source)
    expected = decode_audio(str(path), sampling_rate=16000)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)


def test_path_transcription_decodes_before_model(tmp_path):
    import soundfile as sf

    path = tmp_path / "source.wav"
    sf.write(path, np.sin(np.linspace(0, 60, 8000)) * .1, 8000)
    model = FakeModel([segment()])
    result = transcribe_path(model, path)
    assert result["text"] == "Actual words"
    assert len(model.calls[0][0]) == 16000
