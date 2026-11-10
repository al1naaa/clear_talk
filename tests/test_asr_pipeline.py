"""Unit tests for the Whisper fine-tuning pipeline (run with .venv-train)."""

import json

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

from cleartalk_asr import augment, dataset, mixtures, splits  # noqa: E402
from cleartalk_asr.paths import BASE_WHISPER, ROOT  # noqa: E402
from cleartalk_asr.text import corpus_wer, edit_counts, normalize, pair_counts  # noqa: E402

needs_base = pytest.mark.skipif(not (BASE_WHISPER / "normalizer.json").is_file(), reason="Whisper files not downloaded")


@needs_base
def test_whisper_normalizer_ignores_formatting_only():
    assert normalize("Mr. Hardy's colour, 1920!") == normalize("MISTER HARDY'S COLOR NINETEEN TWENTY")
    assert normalize("Hello, world", "legacy") == "hello world"


@needs_base
def test_edit_counts_and_pooled_wer():
    counts = edit_counts("the cat sat", "the cat sat down")
    assert (counts["insertions"], counts["reference_words"]) == (1, 3)
    assert edit_counts("a b", "")["deletions"] == 2
    # Pooled, not averaged: (0 + 2) / (3 + 1) = 0.5, while the mean of per-utterance rates is 1.0.
    assert corpus_wer(["the cat sat", "dog"], ["the cat sat", "a frog"]) == pytest.approx(0.5)


@needs_base
def test_pair_counts_uses_cheaper_track_assignment():
    result = pair_counts(["red green blue", "cat dog"], ["cat dog", "red green blue"])
    assert result["errors"] == 0 and result["order"] == [1, 0]


@needs_base
def test_target_tokens_formats():
    tokenizer = dataset.processor().tokenizer
    plain = dataset.target_tokens("Hello world.")
    assert tokenizer.convert_ids_to_tokens(plain[:2]) == ["<|startoftranscript|>", "<|notimestamps|>"]
    assert plain[-1] == tokenizer.convert_tokens_to_ids("<|endoftext|>")
    timed = dataset.target_tokens("Hello world.", span=[0.31, 2.0])
    begin = tokenizer.convert_tokens_to_ids("<|0.00|>")
    assert timed[1] == begin + round(0.31 / 0.02) and timed[-2] == begin + 100


@needs_base
def test_collate_shifts_labels_and_masks_padding():
    batch = [{"input_features": np.zeros((80, 3000), np.float32), "tokens": [1, 2, 3, 4]},
             {"input_features": np.zeros((80, 3000), np.float32), "tokens": [1, 5, 6]}]
    out = dataset.collate(batch)
    assert out["decoder_input_ids"].tolist()[0] == [1, 2, 3]
    assert out["labels"].tolist() == [[2, 3, 4], [5, 6, -100]]
    assert out["input_features"].shape == (2, 80, 3000)


def test_timestamp_subset_is_stable():
    assert dataset.uses_timestamps("train-pair-00001-0") == dataset.uses_timestamps("train-pair-00001-0")
    share = np.mean([dataset.uses_timestamps(f"t{i}") for i in range(4000)])
    assert 0.45 < share < 0.55


def test_render_pair_is_deterministic(tmp_path, monkeypatch):
    rate = augment.RATE
    tone = (0.1 * np.sin(np.linspace(0, 400, rate * 2))).astype(np.float32)
    import soundfile as sf

    for name in ("a.flac", "b.flac"):
        sf.write(tmp_path / name, tone, rate)
    monkeypatch.setattr(augment, "ROOT", tmp_path)
    recipe = {"sources": [{"path": "a.flac"}, {"path": "b.flac"}], "offsets_seconds": [0.0, 0.5],
              "gains_db": [0.0, -3.0], "snr_db": 5.0, "noise": "pink", "noise_path": None, "reverb": True, "seed": 7}
    first, clean, spans = augment.render(recipe)
    second, _, _ = augment.render(recipe)
    assert first.dtype == np.int16 and np.array_equal(first, second)
    assert clean.shape == (len(first), 2) and len(first) == int(2.5 * rate)
    assert spans[1][0] >= 0.5


def test_si_sdr_prefers_matching_source():
    rng = np.random.default_rng(0)
    target = rng.standard_normal(8000)
    assert mixtures.si_sdr(target + 0.01 * rng.standard_normal(8000), target) > 30
    assert mixtures.si_sdr(rng.standard_normal(8000), target) < 0


def test_source_fields_never_overwrite_track_path():
    fields = mixtures._source_fields({"path": "data/asr/LibriSpeech/x.flac", "utt_id": "1-2-3"})
    assert "path" not in fields and fields["source_path"].endswith("x.flac")


def test_unit_wer_handles_pairs_and_singles():
    sep = splits.SEP
    refs = [f"red green{sep}blue black", "cat dog"]
    hyps = [f"blue black{sep}red green", "cat"]
    if not (BASE_WHISPER / "normalizer.json").is_file():
        pytest.skip("Whisper files not downloaded")
    assert splits.unit_wer(refs, hyps) == pytest.approx(1 / 6)


def test_plans_are_speaker_disjoint_if_present():
    plans = ROOT / "data/asr/plans"
    if not (plans / "train.jsonl").is_file():
        pytest.skip("plans not generated")
    speakers = {}
    for name in ("train", "val", "test"):
        speakers[name] = {s["speaker"] for line in (plans / f"{name}.jsonl").read_text().splitlines()
                          for s in json.loads(line)["sources"]}
    assert not speakers["train"] & speakers["val"]
    assert not speakers["train"] & speakers["test"]
    assert not speakers["val"] & speakers["test"]
    assert not speakers["train"] & {"198", "5703"}


def test_learning_rate_schedule_warms_up_then_decays_to_floor():
    from cleartalk_asr.train import lr_factor

    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100) == pytest.approx(0.1)
    values = [lr_factor(s, 10, 100) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_epoch_batches_cover_each_example_once_and_are_reproducible():
    from cleartalk_asr.train import epoch_batches

    first = epoch_batches(103, 4, seed=1, epoch=0)
    assert first == epoch_batches(103, 4, seed=1, epoch=0)
    assert first != epoch_batches(103, 4, seed=1, epoch=1)
    flat = [i for batch in first for i in batch]
    assert len(flat) == len(set(flat)) == 100 and all(len(b) == 4 for b in first)


def test_training_tracks_drop_failed_separations_only(monkeypatch):
    import pandas as pd

    from cleartalk_asr import train

    frame = pd.DataFrame({"track_id": ["a", "b", "c"], "recipe_id": ["train-pair-00001", "train-pair-00001", "train-single-00001"],
                          "kind": ["pair", "pair", "single"], "si_sdr_db": [-6.0, 4.0, None]})
    monkeypatch.setattr(train, "load_tracks", lambda name: frame)
    kept = train.training_tracks()
    assert list(kept.track_id) == ["b", "c"]


def _fake_details(mixture_errors, clean_errors):
    import pandas as pd

    def details(set_name, label):
        errors = mixture_errors[label]
        units = pd.DataFrame({"recipe_id": [f"m{i}" for i in range(len(errors))], "kind": "pair",
                              "errors": errors, "reference_words": 100})
        metrics = {"pairs": {"wer": sum(errors) / (100 * len(errors)), "mixtures": len(errors),
                             "reference_words": 100 * len(errors)},
                   "singles": {"wer": clean_errors[label] / 1000, "utterances": 10}}
        return metrics, units
    return details


@pytest.mark.parametrize(("tuned", "clean_tuned", "promoted", "failed_gate"), [
    ([20] * 40, 30, True, None),                                   # 33% better everywhere, clean unchanged
    ([29] * 40, 30, False, "relative_wer_reduction_at_least_5_percent"),
    ([5] * 20 + [50] * 20, 30, False, "bootstrap_95_interval_below_zero"),  # better on average, not reliably
    ([20] * 40, 36, False, "clean_speech_not_worse_than_0.5_points"),
])
def test_promotion_rule_needs_all_three_gates(monkeypatch, tuned, clean_tuned, promoted, failed_gate):
    import importlib.util

    spec = importlib.util.spec_from_file_location("activate_model", ROOT / "scripts/activate_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = [30] * 20 + [31] * 20 if failed_gate == "bootstrap_95_interval_below_zero" else [30] * 40
    monkeypatch.setattr(module, "details", _fake_details({"baseline": baseline, "finetuned": tuned},
                                                         {"baseline": 30, "finetuned": clean_tuned}))
    decision = module.decide()
    assert decision["promoted"] is promoted
    if failed_gate:
        assert decision["gates"][failed_gate] is False


def test_ami_window_helpers():
    from cleartalk_asr import ami

    assert ami._merge([(0, 1), (1.1, 2), (5, 6)], gap=0.2) == [[0, 2], [5, 6]]
    assert ami._overlap([[0, 4]], [[1, 2], [3, 10]]) == pytest.approx(2.0)
    words = {"A": [(1.0 + i, 1.8 + i, f"a{i}") for i in range(8)],
             "B": [(1.5 + i, 2.3 + i, f"b{i}") for i in range(8)], "C": [(40.0, 40.5, "far")], "D": []}
    windows = ami.candidate_windows(words)
    assert windows and all(len(w["speakers"]) == 2 and w["overlap_seconds"] >= 1.0 for w in windows)
    picked = ami.choose(windows, 3)
    assert all(a["end"] <= b["start"] for a, b in zip(picked, picked[1:]))
