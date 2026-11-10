"""Repetition loops are cut back; ordinary repetition stays."""
from backend.loops import cut_loops
from cleartalk_ml.evaluate import without_loops


def test_a_word_written_dozens_of_times_is_cut_back_to_three():
    text, cut = cut_loops("welcome to the Senate " + "Senate " * 40)
    assert cut and text == "welcome to the Senate Senate Senate"


def test_a_repeated_phrase_is_cut_back_and_the_rest_is_kept():
    assert cut_loops("я не знаю, " * 6 + "вот") == ("я не знаю, я не знаю, я не знаю, вот", True)


def test_ordinary_repetition_is_left_alone():
    for text in ("да да да, конечно", "Мен мен мен барамын", "- - - - -", ""):
        assert cut_loops(text) == (text, False)


def test_stored_transcripts_are_compared_as_the_app_now_writes_them():
    rows = {"a": {"turns": [{"text": "иә " * 8, "speaker": "Speaker 1"}, {"text": "жоқ", "speaker": "Speaker 2"}], "auto": "", "forced": ""},
            "b": {"auto": "go go go go go", "forced": "go go go go go"}}
    out = without_loops(rows)
    assert out["a"]["auto"] == "иә иә иә жоқ" and out["b"]["forced"] == "go go go"
    assert without_loops(out) == out


def test_the_old_app_can_be_measured_without_the_loop_cut(tmp_path, monkeypatch):
    import numpy as np

    from backend import conversation, speakers
    from tests.test_conversation import make_pipeline

    pipeline = make_pipeline(tmp_path, monkeypatch, [{"language": "en", "uncertain": False}] * 2)
    monkeypatch.setattr(conversation, "diarize", lambda audio, encoder, count=None: [speakers.Turn(0.0, 3.0, 0)])
    loop = "Senate " * 10
    monkeypatch.setattr(conversation, "transcribe_window", lambda model, audio, language: {
        "text": loop.strip(), "language": language, "speech_seconds": 3.0, "segments": [{"start": 0.0, "end": 3.0, "text": loop.strip(), "needs_review": False}]})
    audio = np.zeros(16000 * 4, dtype=np.float32)
    assert pipeline.transcribe(audio, language="en")["turns"][0]["text"] == "Senate Senate Senate"
    assert pipeline.transcribe(audio, language="en", cut_repeats=False)["turns"][0]["text"] == loop.strip()
