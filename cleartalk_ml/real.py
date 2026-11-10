"""Score the app on real recordings whose transcripts a person corrected (the local check examples).

    python -m cleartalk_ml.real                      # every ready way of transcribing, on every saved example
    python -m cleartalk_ml.real --system auto        # only the app's default (language detected turn by turn)

How an example gets here: transcribe a recording in the app's Conversation mode, press "Correct transcript",
fix the text, speakers and languages, and press "Save as check example". The audio and the corrected turns
are stored in ``data/real/<id>/`` on this computer only (not versioned, not packaged).

Systems (all through the app's own Conversation pipeline, so the number is about what the user gets):

* ``auto``            - language detected per turn, each language to its routed recognizer;
* ``mixed``           - every turn to the recognizer trained for language changes inside a turn;
* ``<recognizer>``    - one recognizer for everything (e.g. ``cleartalk``, ``cleartalk-large``, ``turbo``).

Scores: pooled WER and CER over all examples, and errors per language of the corrected turns. Speaker labels
are not scored here. These recordings were SEEN by a person while correcting the app's own output, so they
are a check set, not a blind test: use them to compare systems, not to claim an absolute accuracy.
"""

import argparse
import json

import pandas as pd

from .paths import LANGUAGES, ROOT
from .text import counts, pooled, segment_counts

EXAMPLES = ROOT / "data" / "real"


def examples():
    rows = []
    for folder in sorted(EXAMPLES.glob("*")):
        reference = folder / "reference.json"
        if reference.is_file() and (folder / "audio.wav").is_file():
            rows.append((folder, json.loads(reference.read_text(encoding="utf-8"))))
    return rows


def transcribe(pipeline, audio, system):
    if system == "auto":
        result = pipeline.transcribe(audio, language="auto", model="auto")
    elif system == "mixed":
        result = pipeline.transcribe(audio, language="mixed", model="auto")
    else:
        result = pipeline.transcribe(audio, language="auto", model=system)
    return [{"start": t["start"], "end": t["end"], "speaker": t["speaker"], "language": t["language"],
             "recognizer": t["recognizer"], "text": t["text"]} for t in result["turns"]]


def run(systems):
    import soundfile as sf

    from backend.config import Settings
    from backend.conversation import ConversationPipeline

    from .awake import keep_awake

    keep_awake()
    settings = Settings()
    status = settings.multilingual_status()
    pipeline = ConversationPipeline(settings)
    rows, by_language = [], {}
    for folder, reference in examples():
        audio, rate = sf.read(folder / "audio.wav", dtype="float32")
        text = " ".join(turn["text"] for turn in reference["turns"])
        segments = [(turn["language"], turn["text"]) for turn in reference["turns"]]
        for system in systems:
            if system not in ("auto", "mixed") and not status["recognizers"].get(system, {}).get("available"):
                continue
            cache = folder / f"hyp-{system}.json"
            if cache.is_file():
                turns = json.loads(cache.read_text(encoding="utf-8"))
            else:
                turns = transcribe(pipeline, audio, system)
                cache.write_text(json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8")
            hypothesis = " ".join(turn["text"] for turn in turns)
            rows.append({"example": folder.name[:8], "system": system, "seconds": round(len(audio) / rate, 1),
                         **counts(text, hypothesis)})
            for language, block in segment_counts(segments, hypothesis).items():
                total = by_language.setdefault((system, language), {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0})
                for key, value in block.items():
                    total[key] += value
    return pd.DataFrame(rows), by_language


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", action="append", help="auto, mixed, or a recognizer of the app profile; repeatable")
    args = parser.parse_args()
    if not examples():
        raise SystemExit(f"No check examples in {EXAMPLES} yet: save some from the app first (see this file's header).")
    systems = args.system or ["auto", "mixed", "cleartalk", "cleartalk-large", "turbo", "small"]
    frame, by_language = run(systems)
    summary = []
    for system, group in frame.groupby("system", sort=False):
        total = pooled(group.to_dict("records"))
        row = {"system": system, "examples": len(group), "minutes": round(group.seconds.sum() / 60, 1), "words": total["reference_words"],
               "WER %": round(100 * total["wer"], 1), "CER %": round(100 * total["cer"], 1)}
        for language in (*LANGUAGES, "mixed"):
            block = by_language.get((system, language))
            if block and block["reference_words"]:
                errors = block["substitutions"] + block["deletions"] + block["insertions"]
                row[f"WER {language} %"] = round(100 * errors / block["reference_words"], 1)
        summary.append(row)
    table = pd.DataFrame(summary)
    print(table.to_string(index=False))
    (EXAMPLES / "scores.json").write_text(json.dumps({"summary": summary, "per_example": frame.to_dict("records")}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
