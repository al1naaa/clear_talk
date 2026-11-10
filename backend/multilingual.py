"""Shared multilingual recognition policy: Kazakh, Russian and English.

Serving and evaluation both call these functions, so a number measured during development is a
number about the app. The policy mirrors the English one (beam 5, speech detection, no carry-over
of earlier text, one temperature) and adds the language choice:

* a fixed language chosen by the user, or
* automatic detection restricted to the three supported languages - Whisper scores about a hundred
  languages, and letting it pick Ukrainian or Kyrgyz for a short Russian or Kazakh phrase would
  switch the whole transcript to the wrong language.
"""

import numpy as np

from .transcription import SAMPLE_RATE, VAD_PARAMETERS

LANGUAGES = ("kk", "ru", "en")
LANGUAGE_NAMES = {"kk": "Kazakh", "ru": "Russian", "en": "English", "mixed": "Mixed"}
# Recognizers trained in this project to write what they hear whatever language token they are given:
# for their turns the language is read from the text, and they may serve turns that mix languages.
WRITES_WHAT_IT_HEARS = ("cleartalk", "cleartalk-large", "cleartalk-kk")
POLICY_ID = "multilingual-vad"
# Below this share of the three-language probability the detection is reported as uncertain.
UNCERTAIN_LANGUAGE = 0.6
MIN_DETECTION_SECONDS = 1.0


def decoding_policy(language) -> dict:
    """Return a fresh dictionary so callers cannot mutate the shared policy.

    ``language`` None lets the recognizer choose its language token itself (only for the recognizers trained
    to write what they hear, whose turn labels are read from the text).
    """
    if language is not None and language not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}")
    return {
        "language": language,
        "task": "transcribe",
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": dict(VAD_PARAMETERS),
        "condition_on_previous_text": False,
        "temperature": 0.0,
    }


def restricted_language(all_language_probs) -> tuple[str, float, dict]:
    """Pick the most probable supported language; return it, its share and all three shares."""
    scores = {language: 0.0 for language in LANGUAGES}
    for language, probability in all_language_probs or []:
        if language in scores:
            scores[language] = float(probability)
    total = sum(scores.values())
    if total <= 0:
        return "en", 0.0, scores
    shares = {language: value / total for language, value in scores.items()}
    best = max(LANGUAGES, key=lambda language: shares[language])
    return best, shares[best], shares


def detect_language(model, audio) -> dict:
    """Detect the language of mono 16 kHz audio among the supported three."""
    from faster_whisper.vad import VadOptions

    audio = np.asarray(audio, dtype=np.float32)
    raw, raw_probability, all_probs = model.detect_language(
        audio, vad_filter=True, vad_parameters=VadOptions(**VAD_PARAMETERS))
    language, share, shares = restricted_language(all_probs)
    return {"language": language, "share": round(share, 4), "shares": {k: round(v, 4) for k, v in shares.items()},
            "unrestricted": raw, "unrestricted_probability": round(float(raw_probability), 4),
            "uncertain": share < UNCERTAIN_LANGUAGE or len(audio) < MIN_DETECTION_SECONDS * SAMPLE_RATE}


_KAZAKH_ONLY = set("әғқңөұүһі")


def language_of_text(text: str, detected: str) -> str:
    """The language a transcript is written in, judged by its letters.

    Used for turns written by the ClearTalk multilingual recognizer: it was trained to write what it
    hears whatever language token it is given, so its text is a better witness of the language than
    the token. On validation this named Kazakh correctly for 100% (FLEURS) and 97% (KSC2) of utterances,
    against 74% and 69% for the model's own detection; Russian and English stayed at 99.7-100%.
    Mostly Latin letters -> English; any Kazakh-only letter -> Kazakh; other Cyrillic -> Russian,
    except that a very short phrase detected as Kazakh stays Kazakh.
    """
    counts = {"lat": 0, "kk": 0, "cyr": 0}
    for word in text.lower().split():
        letters = [ch for ch in word if ch.isalpha()]
        if not letters:
            continue
        if _KAZAKH_ONLY & set(letters):
            counts["kk"] += 1
        elif any("а" <= ch <= "я" or ch == "ё" for ch in letters):
            counts["cyr"] += 1
        elif any("a" <= ch <= "z" for ch in letters):
            counts["lat"] += 1
    total = sum(counts.values())
    if not total:
        return detected
    if counts["lat"] > counts["kk"] + counts["cyr"]:
        return "en"
    if counts["kk"]:
        return "kk"
    return "kk" if detected == "kk" and total < 4 else "ru"


def languages_of_text(text: str, detected: str) -> list[str]:
    """Languages a transcript is written in, in the order they first appear; several when alphabets are mixed.

    A language counts as present when at least two words and 15% of the words are in its alphabet.
    Cyrillic words are Kazakh when the text has Kazakh-only letters, otherwise Russian (a turn that mixes
    Kazakh and Russian cannot be told apart by letters alone and is labelled by the more specific one).
    """
    counts = {"lat": 0, "kk": 0, "cyr": 0}
    first = {}
    for position, word in enumerate(text.lower().split()):
        letters = [ch for ch in word if ch.isalpha()]
        if not letters:
            continue
        if _KAZAKH_ONLY & set(letters):
            kind = "kk"
        elif any("а" <= ch <= "я" or ch == "ё" for ch in letters):
            kind = "cyr"
        elif any("a" <= ch <= "z" for ch in letters):
            kind = "lat"
        else:
            continue
        counts[kind] += 1
        first.setdefault("lat" if kind == "lat" else "cyrillic", position)
    total = sum(counts.values())
    if not total:
        return [detected]
    words = {"lat": counts["lat"], "cyrillic": counts["kk"] + counts["cyr"]}
    name = {"lat": "en", "cyrillic": "kk" if counts["kk"] else "ru"}
    present = [name[kind] for kind in sorted(first, key=first.get) if words[kind] >= 2 and words[kind] / total >= 0.15]
    return present if len(present) > 1 else [language_of_text(text, detected)]


def transcribe_window(model, audio, language: str) -> dict:
    """Transcribe mono 16 kHz audio in one known language; returns text, timed segments and scores."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("ASR input must be a finite mono waveform at 16 kHz")
    if audio.size == 0 or float(np.max(np.abs(audio))) <= 1e-8:
        return {"text": "", "segments": [], "speech_seconds": 0.0, "language": language}
    segments, info = model.transcribe(audio, **decoding_policy(language))
    items = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            items.append({"start": round(float(segment.start), 2), "end": round(float(segment.end), 2), "text": text,
                          "needs_review": bool(segment.avg_logprob < -0.9 or segment.no_speech_prob > 0.6)})
    if language is None:  # the token the recognizer chose, when it is one of the three
        language = info.language if info.language in LANGUAGES else None
    return {"text": " ".join(item["text"] for item in items), "segments": items,
            "speech_seconds": round(float(info.duration_after_vad), 3), "language": language}
