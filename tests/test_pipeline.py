"""The selected recognizer must transcribe every track; deterministic doubles, no real models."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from backend.config import Settings
from backend.pipeline import AudioPipeline


class Separator:
    def separate_batch(self, mixture):
        return torch.stack((mixture, -0.5 * mixture), dim=-1)


class FixturePipeline(AudioPipeline):
    def __init__(self, settings):
        super().__init__(settings)
        self.calls = []

    def _load_asr(self, model):
        self._recognizers[model] = object()
        return self._recognizers[model]

    def _load_separator(self):
        return Separator()

    def _transcribe(self, model, path, duration):
        self.calls.append((model, Path(path).name))
        return {"text": f"{model} text", "segments": [], "review_notes": []}


def job(root, model, mode="separate"):
    directory = root / f"job-{model}-{mode}"
    directory.mkdir()
    wave = 0.1 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)
    sf.write(directory / "original.wav", wave, 16000)
    (directory / "input.json").write_text(json.dumps({"model": model, "is_demo": False}))
    return directory


class PipelineRoutingTests(unittest.TestCase):
    def test_selected_recognizer_transcribes_original_and_both_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for model in ("baseline", "cleartalk"):
                pipeline = FixturePipeline(Settings(root=root))
                result = pipeline.process("fixture", job(root, model), "separate", lambda *_: None)
                self.assertEqual(pipeline.calls, [(model, "original.wav"), (model, "source-a.wav"), (model, "source-b.wav")])
                self.assertEqual(result["model"], model)
                self.assertIn(Settings(root=root).recognizer_name(model), result["models"]["asr"])
                self.assertEqual([s["text"] for s in result["sources"]], [f"{model} text"] * 2)
                self.assertTrue(pipeline.inference_verified)

    def test_transcribe_mode_uses_the_selected_recognizer_without_separation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline = FixturePipeline(Settings(root=root))
            result = pipeline.process("fixture", job(root, "cleartalk", "transcribe"), "transcribe", lambda *_: None)
            self.assertEqual(pipeline.calls, [("cleartalk", "original.wav")])
            self.assertEqual(result["sources"], [])
            self.assertIsNone(result["models"]["separator"])

    def test_unknown_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "Unsupported speech recognizer"):
                FixturePipeline(Settings(root=root)).process("fixture", job(root, "custom"), "separate", lambda *_: None)

    def test_missing_files_fail_before_any_weights_load(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = AudioPipeline(Settings(root=Path(temp)))
            with self.assertRaisesRegex(RuntimeError, "Missing model files"):
                pipeline._load_asr("cleartalk")
            self.assertFalse(pipeline.loaded_models()["cleartalk"])


if __name__ == "__main__":
    unittest.main()
