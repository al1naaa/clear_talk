"""Build the Conversation-mode demo: six short utterances, three languages, six different readers.

    python scripts/prepare_multilingual_demo.py

The utterances come from the FLEURS training part (CC BY 4.0) and are chosen by a fixed seed, not by
how well any model transcribes them. Their ids are written to data/ml/plans/demo-utterances.json and
the cleaning step removes exactly these utterances from training, so the demo stays an example the
project's own model has never been trained on. Output: data/demo/conversation.wav + conversation.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 2026101306
ORDER = (("kk", "female"), ("ru", "male"), ("en", "female"), ("kk", "male"), ("ru", "female"), ("en", "male"))
SECONDS = (4.0, 8.0)
GAP_SECONDS = 0.45
PLAN = ROOT / "data" / "ml" / "plans" / "demo-utterances.json"


def choose():
    from cleartalk_ml.corpora import load_manifest

    table = load_manifest()
    pool = table[(table.corpus == "fleurs") & (table.split == "train") & table.duration.between(*SECONDS)]
    rng = np.random.default_rng(SEED)
    rows, sentences = [], set()
    for language, gender in ORDER:
        candidates = pool[(pool.language == language) & (pool.gender == gender) & ~pool.sentence_id.isin(sentences)]
        row = candidates.sort_values("utt_id").iloc[int(rng.integers(len(candidates)))]
        sentences.add(row.sentence_id)
        rows.append({"utt_id": row.utt_id, "language": language, "gender": gender, "path": row.path,
                     "duration": round(float(row.duration), 2), "text": row.text_raw})
    return rows


def main():
    import soundfile as sf

    from cleartalk_ml.corpora import read_audio

    if PLAN.is_file():
        rows = json.loads(PLAN.read_text(encoding="utf-8"))["utterances"]
    else:
        rows = choose()
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps({"seed": SEED, "utterances": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    pieces, parts, cursor = [], [], 0.0
    for row in rows:
        audio = read_audio(row["path"]).astype(np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        audio = audio * (10 ** (-23 / 20) / max(rms, 1e-6))
        audio = audio * min(1.0, 0.95 / max(float(np.abs(audio).max()), 1e-6))
        parts.append({"start": round(cursor, 2), "end": round(cursor + len(audio) / 16000, 2), "language": row["language"],
                      "reader": row["gender"], "text": row["text"]})
        pieces += [audio, np.zeros(int(GAP_SECONDS * 16000), dtype=np.float32)]
        cursor += len(audio) / 16000 + GAP_SECONDS
    demo = ROOT / "data" / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    sf.write(demo / "conversation.wav", np.concatenate(pieces[:-1]), 16000, subtype="PCM_16")
    (demo / "conversation.json").write_text(json.dumps({
        "title": "Three languages, six readers",
        "description": "Six read sentences in Kazakh, Russian and English placed one after another. An assembled "
                       "demonstration, not a natural conversation.",
        "source": "FLEURS (google/fleurs), training part, CC BY 4.0",
        "parts": parts}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{demo / 'conversation.wav'} · {cursor - GAP_SECONDS:.1f} s · {len(rows)} utterances")


if __name__ == "__main__":
    main()
