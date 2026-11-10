"""Text normalization and error counting for Kazakh, Russian and English.

One rule for all three languages, so numbers are comparable: Unicode NFKC, lowercase, "ё" -> "е",
punctuation and symbols removed, apostrophes kept only inside words. Nothing language-specific is
rewritten (no number spelling, no contraction expansion), so a recognizer is not rewarded or
punished for formatting choices beyond case and punctuation.

WER = (substitutions + deletions + insertions) / reference words, pooled over a set.
CER is the same over characters (spaces removed); it is reported next to WER because Kazakh is
agglutinative: one wrong suffix makes a whole long word wrong.
"""

import re
import unicodedata

_NOT_WORD = re.compile(r"[^\w\s']", re.UNICODE)
_STRAY_APOSTROPHE = re.compile(r"(?<![^\W\d_])'|'(?![^\W\d_])", re.UNICODE)
KAZAKH_ONLY = set("әғқңөұүһі")
CYRILLIC = re.compile(r"[а-яёәғқңөұүһі]")
LATIN = re.compile(r"[a-z]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("ё", "е")
    text = text.replace("’", "'").replace("`", "'").replace("_", " ")
    text = _NOT_WORD.sub(" ", text)
    text = _STRAY_APOSTROPHE.sub(" ", text)
    return " ".join(text.split())


def script_of(word: str) -> str:
    """'kk' for a word with Kazakh-only letters, 'cyr' for other Cyrillic, 'lat' for Latin, else 'other'."""
    word = word.lower()
    if KAZAKH_ONLY & set(word):
        return "kk"
    if CYRILLIC.search(word):
        return "cyr"
    return "lat" if LATIN.search(word) else "other"


def word_counts(reference: str, hypothesis: str) -> dict:
    """Substitutions, deletions, insertions and reference length (in words) for one pair."""
    from jiwer import process_words

    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return {"substitutions": 0, "deletions": 0, "insertions": len(hyp.split()), "reference_words": 0}
    if not hyp:
        words = len(ref.split())
        return {"substitutions": 0, "deletions": words, "insertions": 0, "reference_words": words}
    out = process_words(ref, hyp)
    return {"substitutions": out.substitutions, "deletions": out.deletions, "insertions": out.insertions,
            "reference_words": out.substitutions + out.deletions + out.hits}


def char_counts(reference: str, hypothesis: str) -> dict:
    """Character edits (spaces removed) and reference length in characters."""
    from jiwer import process_characters

    ref, hyp = normalize(reference).replace(" ", ""), normalize(hypothesis).replace(" ", "")
    if not ref:
        return {"char_errors": len(hyp), "reference_chars": 0}
    if not hyp:
        return {"char_errors": len(ref), "reference_chars": len(ref)}
    out = process_characters(ref, hyp)
    return {"char_errors": out.substitutions + out.deletions + out.insertions,
            "reference_chars": out.substitutions + out.deletions + out.hits}


def counts(reference: str, hypothesis: str) -> dict:
    return {**word_counts(reference, hypothesis), **char_counts(reference, hypothesis)}


def pooled(rows) -> dict:
    """Pooled WER and CER over rows produced by ``counts`` (never a mean of per-utterance rates)."""
    keys = ("substitutions", "deletions", "insertions", "reference_words", "char_errors", "reference_chars")
    total = {key: int(sum(row.get(key, 0) for row in rows)) for key in keys}
    total["errors"] = total["substitutions"] + total["deletions"] + total["insertions"]
    total["wer"] = total["errors"] / max(total["reference_words"], 1)
    total["cer"] = total["char_errors"] / max(total["reference_chars"], 1)
    return total


def segment_counts(segments, hypothesis: str) -> dict:
    """Errors per language inside one mixed-language recording.

    ``segments`` is a list of (language, reference text) in spoken order. The hypothesis is aligned
    to the concatenated reference once; every substitution and deletion belongs to the language of
    its reference word, and an insertion to the language of the reference word just before it.
    """
    from jiwer import process_words

    words, owner = [], []
    for language, text in segments:
        for word in normalize(text).split():
            words.append(word)
            owner.append(language)
    result = {language: {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0}
              for language in dict.fromkeys(owner)}
    for language in owner:
        result[language]["reference_words"] += 1
    hyp = normalize(hypothesis)
    if not words:
        return result
    if not hyp:
        for language in owner:
            result[language]["deletions"] += 1
        return result
    out = process_words(" ".join(words), hyp)
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            continue
        if chunk.type == "insert":
            at = min(max(chunk.ref_start_idx - 1, 0), len(owner) - 1)
            result[owner[at]]["insertions"] += chunk.hyp_end_idx - chunk.hyp_start_idx
            continue
        key = "substitutions" if chunk.type == "substitute" else "deletions"
        for index in range(chunk.ref_start_idx, chunk.ref_end_idx):
            result[owner[index]][key] += 1
    return result
