"""Text normalization and word-error counting.

Two normalizers are reported:

* ``whisper`` - Whisper's EnglishTextNormalizer (numbers, spelling variants,
  contractions). It is the primary metric because it removes formatting
  differences such as "1920" vs "nineteen twenty", so a lower WER reflects
  recognition rather than learning LibriSpeech's spelling conventions.
* ``legacy`` - the v4 project's lowercase/punctuation-only rule, kept so new
  numbers can be compared with the earlier ClearTalk reports.
"""

import itertools
import json
import re
from functools import lru_cache

from .paths import BASE_WHISPER


@lru_cache(maxsize=1)
def _whisper_normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    mapping = json.loads((BASE_WHISPER / "normalizer.json").read_text())
    return EnglishTextNormalizer(mapping)


def normalize(text: str, kind: str = "whisper") -> str:
    if kind == "whisper":
        return " ".join(_whisper_normalizer()(text).split())
    if kind == "legacy":
        return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())
    raise ValueError(f"Unknown normalizer {kind!r}")


def edit_counts(reference: str, hypothesis: str, kind: str = "whisper") -> dict:
    """Substitutions, deletions, insertions and reference length for one pair."""
    from jiwer import process_words

    ref, hyp = normalize(reference, kind), normalize(hypothesis, kind)
    if not ref:
        return {"substitutions": 0, "deletions": 0, "insertions": len(hyp.split()), "reference_words": 0}
    if not hyp:
        words = len(ref.split())
        return {"substitutions": 0, "deletions": words, "insertions": 0, "reference_words": words}
    out = process_words(ref, hyp)
    return {"substitutions": out.substitutions, "deletions": out.deletions,
            "insertions": out.insertions, "reference_words": out.substitutions + out.deletions + out.hits}


def add_counts(rows) -> dict:
    total = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0}
    for row in rows:
        for key in total:
            total[key] += row[key]
    total["errors"] = total["substitutions"] + total["deletions"] + total["insertions"]
    total["wer"] = total["errors"] / max(total["reference_words"], 1)
    return total


def pair_counts(references, hypotheses, kind: str = "whisper") -> dict:
    """Two separated outputs have arbitrary order: keep the assignment with fewer edits."""
    if len(references) != len(hypotheses):
        raise ValueError("Each reference needs exactly one hypothesis track")
    best = None
    for order in itertools.permutations(range(len(hypotheses))):
        counts = add_counts(edit_counts(references[i], hypotheses[j], kind) for i, j in enumerate(order))
        if best is None or counts["errors"] < best["errors"]:
            best = {**counts, "order": list(order)}
    return best


def corpus_wer(references, hypotheses, kind: str = "whisper") -> float:
    """Pooled WER = (S + D + I) / reference words, never a mean of per-utterance rates."""
    return add_counts(edit_counts(r, h, kind) for r, h in zip(references, hypotheses))["wer"]
