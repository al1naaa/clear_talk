"""Playback contract tests; stub inference is not an accuracy measurement."""
import json
import numpy as np
import soundfile as sf
from backend.config import Settings
from backend.conversation import ConversationPipeline
from backend.overlap import Separator


def test_separated_tracks_are_saved_and_attributed(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path)
    pipeline = ConversationPipeline(settings)
    audio = np.ones(16000, dtype=np.float32) * .1
    sf.write(tmp_path / 'original.wav', audio, 16000)
    turns = [dict(start=0, end=1, speaker=f'Speaker {i+1}', text=text,
                  language=lang, languages=[lang], language_name=name, needs_review=True)
             for i, (text, lang, name) in enumerate([('Сәлем', 'kk', 'Kazakh'), ('Привет', 'ru', 'Russian')])]
    monkeypatch.setattr(pipeline, 'transcribe_two_voices', lambda *a: dict(
        turns=turns, speech_seconds=2, recognizers={'small': 2}, speaker_labels=True,
        background_turns=[], two_voices=True, _source_audio=[audio, -audio]))
    result = pipeline.process('example', tmp_path, {'two_voices': True}, lambda *a: None)
    assert result['two_voices'] and result['requested_two_voices']
    assert result['speaker_label_method'] == 'separated_source_tracks'
    assert result['models']['speakers'] is None
    assert result['models']['separator']
    assert [s['text'] for s in result['sources']] == ['Сәлем', 'Привет']
    for filename, sign in [('source-a.wav', 1), ('source-b.wav', -1)]:
        samples, rate = sf.read(tmp_path / filename)
        assert rate == 16000 and len(samples) == len(audio)
        np.testing.assert_allclose(samples, audio * sign, atol=4e-5)
    json.dumps(result)


def test_fallback_does_not_claim_separation(tmp_path, monkeypatch):
    pipeline = ConversationPipeline(Settings(root=tmp_path))
    sf.write(tmp_path / 'original.wav', np.zeros(16000), 16000)
    monkeypatch.setattr(pipeline, 'transcribe_two_voices', lambda *a: dict(
        turns=[], speech_seconds=0, recognizers={}, speaker_labels=False,
        background_turns=[], two_voices=False, separation_status='fallback_original'))
    result = pipeline.process('example', tmp_path, {'two_voices': True}, lambda *a: None)
    assert result['requested_two_voices'] and not result['two_voices']
    assert result['sources'] == [] and result['models']['separator'] is None
    assert not (tmp_path / 'source-a.wav').exists()


def test_invalid_track_intervals_are_rejected():
    tracks = [np.ones(16000), np.ones(16000)]
    for start, end in [(2, 3), (1, 0), (float('nan'), 1)]:
        assert not Separator.own_voice(tracks, 0, start, end)
