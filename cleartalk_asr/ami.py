"""Real overlapping speech from the AMI Meeting Corpus (CC BY 4.0), cut to the app's 20 s limit.

    python -m cleartalk_asr.ami cut        # find two-speaker overlapping windows, write samples/ami/*.wav
    python -m cleartalk_asr.ami evaluate   # pilot check: both recognizers through the app pipeline

Input (downloaded separately, see samples/ami/README.md): per-meeting ``<meeting>.Mix-Headset.wav``
and the manual word annotations ``words/<meeting>.<A-D>.words.xml`` under data/ami/downloads/.
A window is kept when exactly two of the four participants speak in it, their speech overlaps,
and it is bounded by moments where nobody speaks, so no word is cut. References are the
annotated words of each speaker. This is a small real-data check, not a tuned benchmark: the
recognizers were never adapted to AMI, and AMI speech is spontaneous, accented meeting talk.
"""

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .paths import BASE_WHISPER_CT2, EXPORT, ROOT, SEPARATOR

DOWNLOADS = ROOT / "data" / "ami" / "downloads"
SAMPLES = ROOT / "samples" / "ami"
MEETINGS = ("ES2004a", "IS1009a", "IS1009b", "TS3003a", "EN2002a", "EN2002b")  # all from AMI's official test split
MIN_SECONDS, MAX_SECONDS = 8.0, 19.0
MIN_WORDS, MIN_OVERLAP_SECONDS = 5, 1.0
PER_MEETING = 8
WORD = re.compile(r'<w [^>]*starttime="([\d.]+)" endtime="([\d.]+)"([^>]*)>([^<]*)</w>')


def read_words(meeting):
    """{agent: [(start, end, word)]} without punctuation tokens."""
    words = {}
    for path in sorted((DOWNLOADS / "annotations" / "words").glob(f"{meeting}.*.words.xml")):
        agent = path.name.split(".")[1]
        rows = []
        for start, end, attributes, text in WORD.findall(path.read_text(encoding="latin-1")):
            if 'punc="true"' in attributes or not text.strip():
                continue
            rows.append((float(start), float(end), html.unescape(text.strip())))
        words[agent] = rows
    return words


def _merge(intervals, gap):
    merged = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _overlap(a, b):
    total, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def candidate_windows(words):
    """Windows bounded by global silence in which exactly two agents talk and overlap."""
    everyone = _merge([(s, e) for rows in words.values() for s, e, _ in rows], gap=0.35)
    candidates = []
    for first in range(len(everyone)):
        for last in range(first, len(everyone)):
            start, end = everyone[first][0] - 0.25, everyone[last][1] + 0.25
            if end - start > MAX_SECONDS:
                break
            if end - start < MIN_SECONDS:
                continue
            inside = {agent: [(s, e, w) for s, e, w in rows if s >= start and e <= end] for agent, rows in words.items()}
            active = {agent: rows for agent, rows in inside.items() if rows}
            if len(active) != 2 or any(len(rows) < MIN_WORDS for rows in active.values()):
                continue
            speech = [_merge([(s, e) for s, e, _ in rows], gap=0.2) for rows in active.values()]
            overlap = _overlap(*speech)
            if overlap < MIN_OVERLAP_SECONDS:
                continue
            candidates.append({"start": round(max(start, 0.0), 2), "end": round(end, 2), "overlap_seconds": round(overlap, 2),
                               "speakers": [{"agent": agent, "text": " ".join(w for _, _, w in rows), "words": len(rows)}
                                            for agent, rows in sorted(active.items())]})
    return candidates


def choose(candidates, limit):
    """Most-overlapped windows first, never reusing audio."""
    chosen = []
    for candidate in sorted(candidates, key=lambda c: -c["overlap_seconds"] / (c["end"] - c["start"])):
        if all(candidate["end"] <= other["start"] or candidate["start"] >= other["end"] for other in chosen):
            chosen.append(candidate)
        if len(chosen) == limit:
            break
    return sorted(chosen, key=lambda c: c["start"])


def cut():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    manifest = []
    for meeting in MEETINGS:
        audio, rate = sf.read(DOWNLOADS / f"{meeting}.Mix-Headset.wav", dtype="float32")
        if audio.ndim != 1 or rate != 16000:
            raise ValueError(f"Unexpected audio format for {meeting}")
        windows = choose(candidate_windows(read_words(meeting)), PER_MEETING)
        for index, window in enumerate(windows):
            clip = audio[round(window["start"] * rate):round(window["end"] * rate)].copy()
            peak = float(np.max(np.abs(clip)))
            if peak > 1e-4:
                clip *= 0.9 / peak  # headset mixes are quiet; level only, nothing else is changed
            name = f"ami-{meeting}-{index:02d}.wav"
            sf.write(SAMPLES / name, clip, rate, subtype="PCM_16")
            manifest.append({"file": name, "meeting": meeting, "start_seconds": window["start"], "end_seconds": window["end"],
                             "duration_seconds": round(window["end"] - window["start"], 2),
                             "overlap_seconds": window["overlap_seconds"], "speakers": window["speakers"]})
        print(f"{meeting}: {len(windows)} fragments", flush=True)
    (SAMPLES / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(manifest)} fragments in {SAMPLES}")


def evaluate(threads):
    """The app's own steps on each fragment: 16 kHz -> 8 kHz -> SepFormer -> track -> recognizer."""
    import sys
    import tempfile

    import torch
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly
    from speechbrain.inference.separation import SepformerSeparation

    sys.path.insert(0, str(ROOT))
    from backend.transcription import transcribe_path

    from .text import add_counts, pair_counts

    manifest = json.loads((SAMPLES / "manifest.json").read_text())
    torch.set_num_threads(threads)
    separator = SepformerSeparation.from_hparams(source=str(SEPARATOR), savedir=str(SEPARATOR), run_opts={"device": "cpu"})
    separator.mods.eval()
    models = {name: WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=threads, local_files_only=True)
              for name, path in (("baseline", BASE_WHISPER_CT2), ("cleartalk", EXPORT))}
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for item in manifest:
            audio, rate = sf.read(SAMPLES / item["file"], dtype="float32")
            mixture = resample_poly(audio, 8000, rate).astype(np.float32)
            with torch.inference_mode():
                separated = separator.separate_batch(torch.from_numpy(mixture)[None])[0].numpy()
            paths = []
            for k in range(2):
                track = separated[:, k].copy()
                peak = float(np.max(np.abs(track)))
                if peak > 1e-6:
                    track *= 0.95 / peak
                paths.append(Path(scratch) / f"{item['file']}-{k}.wav")
                sf.write(paths[-1], track, 8000, subtype="PCM_16")
            references = [speaker["text"] for speaker in item["speakers"]]
            row = {"file": item["file"], "references": references}
            for name, model in models.items():
                hypotheses = [transcribe_path(model, path)["text"] for path in paths]
                row[name] = {"hypotheses": hypotheses, **pair_counts(references, hypotheses)}
            rows.append(row)
            print(f"{item['file']}: baseline {row['baseline']['errors']} / cleartalk {row['cleartalk']['errors']} "
                  f"errors of {row['baseline']['reference_words']} words", flush=True)
    summary = {name: add_counts(row[name] for row in rows) for name in models}
    out = ROOT / "data" / "asr" / "results" / "ami-pilot.json"
    out.write_text(json.dumps({"fragments": len(rows), "summary": summary, "rows": rows}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cut", "evaluate"))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    cut() if args.command == "cut" else evaluate(args.threads)


if __name__ == "__main__":
    main()
