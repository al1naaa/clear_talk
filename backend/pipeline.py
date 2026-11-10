"""Separation and speech recognition, loaded lazily from verified local files."""

import json
import logging
import time
from pathlib import Path
from typing import Callable

from .config import RECOGNIZERS, Settings
from .conversation import ConversationPipeline
from .transcription import transcribe_path

logger = logging.getLogger(__name__)
Update = Callable[[str, int], None]


class AudioPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._separator = None
        self._recognizers = {}
        self.conversation = ConversationPipeline(settings)
        self.inference_verified = False

    def loaded_models(self):
        return {"separator": self._separator is not None,
                **{model: model in self._recognizers for model in RECOGNIZERS},
                "conversation": self.conversation.loaded_models()}

    def _load_asr(self, model: str):
        if model not in self._recognizers:
            self.settings.verify_recognizer(model)
            from faster_whisper import WhisperModel

            self._recognizers[model] = WhisperModel(
                str(self.settings.recognizer_dir(model)),
                device="cpu",
                compute_type="int8",
                cpu_threads=self.settings.cpu_threads,
                num_workers=1,
                local_files_only=True,
            )
        return self._recognizers[model]

    def _load_separator(self):
        if self._separator is None:
            self.settings.verify_separator()
            import torch
            from speechbrain.inference.separation import SepformerSeparation

            torch.set_num_threads(self.settings.cpu_threads)
            self._separator = SepformerSeparation.from_hparams(
                source=str(self.settings.separator_dir),
                savedir=str(self.settings.separator_dir),
                run_opts={"device": "cpu"},
            )
            self._separator.mods.eval()
        return self._separator

    def _transcribe(self, model: str, path: Path, duration: float) -> dict:
        return transcribe_path(self._load_asr(model), path, duration)

    def process(self, job_id: str, job_dir: Path, mode: str, update: Update) -> dict:
        import numpy as np
        import soundfile as sf

        started = time.perf_counter()
        context_path = job_dir / "input.json"
        context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.is_file() else {}
        if mode == "conversation":  # multilingual, any number of speakers, long recordings
            result = self.conversation.process(job_id, job_dir, context, update)
            self.inference_verified = True
            return result
        model = context.get("model", self.settings.default_model)
        if model not in RECOGNIZERS:
            raise ValueError("Unsupported speech recognizer")
        original_path = job_dir / "original.wav"
        audio, sample_rate = sf.read(original_path, dtype="float32")
        duration = len(audio) / sample_rate
        update("Loading the English speech recognizer", 10)
        self._load_asr(model)
        update("Transcribing the original recording", 20)
        original = self._transcribe(model, original_path, duration)
        original["audio_url"] = f"/media/{job_id}/original.wav"
        sources, metrics = [], []
        warnings = [
            "English speech only. Transcripts can contain errors; check the audio before relying on them.",
            "Segment times are approximate. Review flags use simple score thresholds, not calibrated confidence.",
            "Speech detection reduces text on non-speech audio, but can miss very quiet or distorted words.",
        ]
        if model == "cleartalk":
            warnings.append("The ClearTalk recognizer was fine-tuned on synthetic mixtures of read English audiobooks. "
                            "Its measured gains do not guarantee better text on every recording; compare with the pretrained model.")
        if mode == "separate":
            import torch
            from scipy.signal import resample_poly

            update("Loading the two-speaker separation model", 35)
            separator = self._load_separator()
            update("Separating two voices from the noisy mixture", 45)
            audio_8k = resample_poly(audio, 8000, sample_rate).astype(np.float32)
            mixture = torch.from_numpy(audio_8k).unsqueeze(0)
            with torch.inference_mode():
                separated = separator.separate_batch(mixture).detach().cpu().numpy()[0]
            if separated.shape != (len(audio_8k), 2) or not np.isfinite(separated).all():
                raise RuntimeError("The separation model did not produce two finite audio tracks matching the input duration.")

            if context.get("is_demo") is True:
                # These references belong only to the packaged synthetic demo mixture.
                reference_paths = [self.settings.demo_dir / "source1.wav", self.settings.demo_dir / "source2.wav"]
                if all(path.is_file() for path in reference_paths):
                    try:
                        from .metrics import matched_si_sdr_improvement

                        references = []
                        for path in reference_paths:
                            reference, ref_rate = sf.read(path, dtype="float32")
                            if reference.ndim != 1:
                                raise ValueError("Demo clean reference must be mono.")
                            references.append(resample_poly(reference, 8000, ref_rate))
                        comparison = matched_si_sdr_improvement(
                            [separated[:, 0], separated[:, 1]], references, audio_8k)
                        metrics.append({
                            "label": "SI-SDR improvement",
                            "value": round(comparison["mean_improvement_db"], 2),
                            "unit": "dB",
                            "description": "Mean separation improvement over this demo's noisy input, against its known clean sources. Audio quality, not transcription accuracy.",
                            "details": comparison,
                        })
                    except Exception:
                        logger.exception("Could not calculate the demo's reference-based metric")
                        warnings.append("The demo audio quality metric could not be calculated.")

            for index, label in enumerate(("Source A", "Source B")):
                filename = f"source-{'a' if index == 0 else 'b'}.wav"
                source = separated[:, index].astype(np.float32)
                peak = float(np.max(np.abs(source)))
                if peak > 1e-6:
                    source = source * (0.95 / peak)
                source_path = job_dir / filename
                sf.write(source_path, source, 8000, subtype="PCM_16")
                update(f"Transcribing {label}", 65 + index * 15)
                transcription = self._transcribe(model, source_path, duration)
                transcription.update({"label": label, "audio_url": f"/media/{job_id}/{filename}"})
                sources.append(transcription)
            warnings.extend([
                "The separator assumes exactly two simultaneous speakers. A single speaker or more speakers can produce misleading tracks.",
                "Source A and Source B are arbitrary labels for this clip, not identified people.",
                "Playback levels are normalized separately. Separation may leave interference or remove words.",
            ])
        else:
            warnings.append("Transcription-only mode does not separate voices or assign speaker labels.")

        update("Preparing transcripts and audio playback", 95)
        self.inference_verified = True
        return {
            "duration_seconds": round(duration, 2),
            "processing_seconds": round(time.perf_counter() - started, 2),
            "mode": mode,
            "model": model,
            "original": original,
            "sources": sources,
            "metrics": metrics,
            "warnings": warnings,
            "models": {
                "separator": f"{self.settings.separator_id} (8 kHz)" if mode == "separate" else None,
                "asr": f"{self.settings.recognizer_name(model)} · CPU int8 · speech detection",
            },
            "download_url": f"/api/jobs/{job_id}/download",
        }
