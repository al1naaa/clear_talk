"""Project locations shared by data preparation, training, evaluation and export."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "asr"
LIBRISPEECH = DATA / "LibriSpeech"
# dev-other is only partly available locally (528 utterances, 33 speakers from the v4 run).
DEV_OTHER = ROOT / "data" / "v4" / "LibriSpeech" / "dev-other"
PC_MANIFESTS = DATA / "librispeech_pc"
NOISE_MANIFEST = ROOT / "data" / "v4" / "esc10" / "manifest.json"
MIXTURES = DATA / "mixtures"
SEPARATOR = ROOT / "models" / "candidates-v3" / "sepformer-libri2mix"
BASE_WHISPER = ROOT / "models" / "whisper-small.en-hf"
BASE_WHISPER_CT2 = ROOT / "models" / "candidates-v3" / "whisper-small.en"
RUNS = ROOT / "models" / "asr-runs"
EXPORT = ROOT / "models" / "whisper-small.en-cleartalk"
PIPELINE = ROOT / "ml_pipeline_english"

# Official LibriSpeech subsets are speaker-disjoint, so roles inherit that guarantee.
ROLE_OF_SUBSET = {
    "train-clean-100": "train",
    "dev-clean": "val",
    "dev-other": "val",
    "test-clean": "test",
    "test-other": "test",
}
# The packaged demo mixes these two train-clean-100 speakers; keep them unseen.
DEMO_SPEAKERS = {"198", "5703"}
