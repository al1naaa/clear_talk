import numpy as np
import pytest

pytest.importorskip("pandas")
from scripts.pitch_scoring import signal_scores, stream_errors


def test_stream_score_handles_swap_order_time_and_extra_talker():
    turns = [
        {"start": 3, "speaker": "A", "text": "world"},
        {"start": 0, "speaker": "A", "text": "hello"},
        {"start": 2, "speaker": "B", "text": "сәлем"},
        {"start": 4, "speaker": "C", "text": "extra word"},
    ]
    assert stream_errors(["сәлем", "hello world"], turns) == (2, 3)
    assert stream_errors(["сәлем", "hello world"], []) == (3, 3)


def test_si_sdri_compares_against_the_same_unprocessed_mixture():
    rng = np.random.default_rng(22)
    ref = rng.normal(size=(2, 1600))
    mix = ref.sum(axis=0)
    before, after = signal_scores(mix, ref, [mix, mix])
    assert after - before == pytest.approx(0)
    a, b = signal_scores(mix, ref, ref[::-1] * 2)
    assert a == pytest.approx(before)
    assert b > 60
