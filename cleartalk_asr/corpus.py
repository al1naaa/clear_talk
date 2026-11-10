"""One row per LibriSpeech utterance: audio path, speaker, duration and both transcripts.

``text`` is the LibriSpeech-PC version (punctuation and capitalization restored,
CC BY 4.0) and is the training target, so the adapted recognizer keeps writing
readable sentences. ``text_ls`` is the original upper-case corpus transcript and
is used as the reference for scoring when the PC version is missing.
"""

import json
from pathlib import Path

import pandas as pd
import soundfile as sf

from .paths import DEMO_SPEAKERS, DEV_OTHER, LIBRISPEECH, PC_MANIFESTS, ROLE_OF_SUBSET, ROOT


def _speaker_genders() -> dict:
    genders = {}
    for line in (LIBRISPEECH / "SPEAKERS.TXT").read_text().splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        genders[parts[0]] = parts[1]
    return genders


def _pc_texts(subset: str) -> dict:
    texts = {}
    for line in (PC_MANIFESTS / f"{subset}.json").read_text().splitlines():
        record = json.loads(line)
        texts[Path(record["audio_filepath"]).stem] = record["text"]
    return texts


def subset_root(subset: str) -> Path:
    return DEV_OTHER if subset == "dev-other" else LIBRISPEECH / subset


def build_manifest() -> pd.DataFrame:
    genders = _speaker_genders()
    rows = []
    for subset, role in ROLE_OF_SUBSET.items():
        pc = _pc_texts(subset)
        root = subset_root(subset)
        for transcript in sorted(root.glob("*/*/*.trans.txt")):
            for line in transcript.read_text().splitlines():
                utt_id, text_ls = line.split(" ", 1)
                path = transcript.parent / f"{utt_id}.flac"
                if not path.is_file():
                    continue  # dev-other is a partial local copy
                speaker, chapter, _ = utt_id.split("-")
                rows.append({
                    "utt_id": utt_id, "subset": subset, "role": role, "speaker": speaker,
                    "chapter": chapter, "gender": genders.get(speaker, "?"),
                    "duration": sf.info(path).duration,
                    "text": pc.get(utt_id), "text_ls": text_ls,
                    "has_pc": utt_id in pc, "demo_speaker": speaker in DEMO_SPEAKERS,
                    "path": path.relative_to(ROOT).as_posix(),
                })
    frame = pd.DataFrame(rows)
    frame["n_words"] = frame["text_ls"].str.split().str.len()
    frame["words_per_second"] = frame["n_words"] / frame["duration"]
    return frame
