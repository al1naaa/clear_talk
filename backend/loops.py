"""Repetition loops: a recogniser that gets stuck writes the same word or phrase dozens of times
("Senate Senate Senate ..."). Real speech repeats a word or a short phrase a few times at most, so any word or phrase
of up to six words that follows itself more than three times in a row is cut back to three occurrences.

Measured on the validation transcripts of the app (see ml_pipeline/round4/PIPELINE.md): clean recordings were not
changed except one turn that was itself a loop ("я не знаю" six times on a mixed recording).
"""

import re

KEEP = 3
LONGEST = 6


def cut_loops(text: str) -> tuple[str, bool]:
    """The text with runs of more than KEEP repeats cut back to KEEP, and whether anything was cut."""
    words = text.split()
    keys = [re.sub(r"[^\w]", "", word.lower()) for word in words]
    kept_words, kept_keys, cut, i = [], [], False, 0
    while i < len(words):
        for n in range(1, LONGEST + 1):
            unit = keys[i:i + n]
            if (len(unit) == n and any(unit) and len(kept_keys) >= KEEP * n
                    and all(kept_keys[len(kept_keys) - n * (r + 1): len(kept_keys) - n * r] == unit for r in range(KEEP))):
                i, cut = i + n, True
                break
        else:
            kept_words.append(words[i])
            kept_keys.append(keys[i])
            i += 1
    return (" ".join(kept_words) if cut else text), cut
