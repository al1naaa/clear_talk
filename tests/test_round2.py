"""Round 2 bookkeeping: the plan of the second final test, scoring on real check examples, what gets packaged.

These tests use tiny synthetic data and a fake transcriber; they check rules, not model quality.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jiwer")

from cleartalk_ml import final_test_b, real


# ----------------------------------------------------------------------------- final test on test B
def test_every_system_of_the_second_final_test_is_run_exactly_once():
    grouped = [system for group in final_test_b.GROUPS.values() for system in group]
    assert sorted(grouped) == sorted(final_test_b.PLAN)
    # The two processes that share the graphics card hold one model each; the pipelines (several models) run alone.
    assert set(final_test_b.GROUPS["last"]) == {"pipeline:auto", "pipeline:mixed"}
    assert not {"pipeline:auto", "pipeline:mixed"} & set(final_test_b.GROUPS["first"] + final_test_b.GROUPS["second"])


def test_only_our_model_is_scored_with_automatic_language_on_single_utterances():
    assert [system for system, (_, forced_only) in final_test_b.PLAN.items() if not forced_only and not system.startswith("pipeline")] == ["cleartalk-large"]
    units = pd.DataFrame({"unit_id": ["a", "b", "c"], "set": ["fleurs-kk", "mixed", "golos-ru"]})
    assert list(final_test_b.units_for("pipeline:mixed", units).unit_id) == ["b"]
    assert list(final_test_b.units_for("kazakh", units).unit_id) == ["a", "b", "c"]


def test_the_second_final_test_reads_its_gate_from_its_own_log(tmp_path, monkeypatch):
    monkeypatch.setattr(final_test_b.round2, "R2", tmp_path)
    (tmp_path / "PIPELINE.md").write_text("- Gate B: approved 2026-10-13 (x)\n- Gate C: pending\n", encoding="utf-8")
    assert not final_test_b.gate_c_approved()
    (tmp_path / "PIPELINE.md").write_text("- Gate C: approved 2026-10-14 (delegated)\n", encoding="utf-8")
    assert final_test_b.gate_c_approved()


# ----------------------------------------------------------------------------- real check examples
class FakePipeline:
    """Returns the reference of the example with one word changed for 'auto' and unchanged for 'mixed'."""

    def __init__(self, settings):
        pass

    def transcribe(self, audio, language="auto", model="auto"):
        text = {"auto": ["сәлем достар", "привет world friends"], "mixed": ["сәлем достар", "привет всем friends"]}[language]
        return {"turns": [{"start": 0.0, "end": 1.0, "speaker": "Speaker 1", "language": "kk", "recognizer": "x", "text": text[0]},
                          {"start": 1.0, "end": 2.0, "speaker": "Speaker 2", "language": "ru", "recognizer": "x", "text": text[1]}]}


def test_real_examples_are_scored_per_system_and_per_language(tmp_path, monkeypatch, capsys):
    sf = pytest.importorskip("soundfile")
    folder = tmp_path / "0123456789abcdef"
    folder.mkdir()
    sf.write(folder / "audio.wav", np.zeros(32000, dtype="float32"), 16000)
    (folder / "reference.json").write_text(json.dumps({"turns": [
        {"start": 0.0, "end": 1.0, "speaker": "Алия", "language": "kk", "text": "Сәлем, достар!"},
        {"start": 1.0, "end": 2.0, "speaker": "Тимур", "language": "mixed", "text": "Привет всем friends"}]}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "not-an-example").mkdir()
    monkeypatch.setattr(real, "EXAMPLES", tmp_path)
    import backend.conversation

    monkeypatch.setattr(backend.conversation, "ConversationPipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["real", "--system", "auto", "--system", "mixed"])
    real.main()
    scores = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    by_system = {row["system"]: row for row in scores["summary"]}
    assert by_system["auto"]["words"] == 5 and by_system["auto"]["WER %"] == 20.0     # one of five words wrong
    assert by_system["mixed"]["WER %"] == 0.0
    assert by_system["auto"]["WER kk %"] == 0.0 and by_system["auto"]["WER mixed %"] == pytest.approx(33.3)
    assert (folder / "hyp-auto.json").is_file()           # transcripts are kept, a second run does not transcribe again
    assert "auto" in capsys.readouterr().out


def test_scoring_real_examples_says_what_to_do_when_there_are_none(tmp_path, monkeypatch):
    monkeypatch.setattr(real, "EXAMPLES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["real"])
    with pytest.raises(SystemExit, match="No check examples"):
        real.main()


# ----------------------------------------------------------------------------- archives
def test_private_recordings_never_enter_the_source_archive(tmp_path, monkeypatch):
    from scripts import package_source

    private = package_source.ROOT / "data" / "real"
    assert not any("data/real" in pattern for pattern in package_source.INCLUDE)
    packaged = {path.relative_to(package_source.ROOT).as_posix() for path in package_source.files()}
    assert not any(name.startswith("data/real/") or name.startswith("data/jobs/") for name in packaged), private
    assert not any(name.startswith("docs/dev/") for name in packaged)
    assert not {"docs/PRESENTATION_RU.md", "docs/SPEECH_RU.md"} & packaged
    assert {"web/i18n.js", "web/app.js", "ml_pipeline/round2/PIPELINE.md", "cleartalk_ml/final_test_b.py",
            "scripts/select_round2.py", "scripts/night_round2.sh", "data/ml/plans/testB-units.csv"} <= packaged
    assert not any(name.endswith((".wav", ".bin", ".safetensors", ".pt")) and not name.startswith(("data/demo/", "samples/")) for name in packaged)
