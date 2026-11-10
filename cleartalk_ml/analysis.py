"""Error analysis on validation transcripts (step 13): where the errors are and what kind they are."""

import pandas as pd

from .evaluate import details, units_of
from .splits import SEP
from .text import normalize, pooled, script_of, segment_counts
from .validation import hypotheses


def per_unit(set_name, system, kind="auto"):
    units = units_of(set_name)
    rows, _ = hypotheses(set_name, system)
    frame = details(units, rows, kind)
    frame["errors"] = frame.substitutions + frame.deletions + frame.insertions
    return frame.merge(units[["unit_id", "languages", "duration", "reference"]], on="unit_id")


def better_same_worse(set_name, baseline, candidate, kind="auto"):
    a, b = per_unit(set_name, baseline, kind), per_unit(set_name, candidate, kind)
    merged = a.merge(b, on="unit_id", suffixes=("_a", "_b"))
    return {"better": int((merged.errors_b < merged.errors_a).sum()), "same": int((merged.errors_b == merged.errors_a).sum()),
            "worse": int((merged.errors_b > merged.errors_a).sum()), "units": int(len(merged))}


def edit_kinds(set_name, system, kind="auto"):
    frame = per_unit(set_name, system, kind)
    return {key: int(frame[key].sum()) for key in ("substitutions", "deletions", "insertions", "reference_words")}


def by_position(system):
    """WER of the 1st, 2nd and later utterances inside mixed validation recordings."""
    units = units_of("mixed-val")
    rows, _ = hypotheses("mixed-val", system)
    total = {}
    for unit in units.itertuples():
        segments = [(f"part {min(index, 3)}{'+' if index >= 3 else ''}", text) for index, text in enumerate(unit.reference.split(SEP), 1)]
        for key, block in segment_counts(segments, rows[unit.unit_id].get("auto", "")).items():
            entry = total.setdefault(key, {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0})
            for name, value in block.items():
                entry[name] += value
    return {key: (v["substitutions"] + v["deletions"] + v["insertions"]) / max(v["reference_words"], 1) for key, v in sorted(total.items())}


def wrong_script(system):
    """Share of reference words answered in the other alphabet (a translation or transliteration).

    English words written in Cyrillic, and Kazakh/Russian words written in Latin letters, counted over the
    substitutions of the alignment in mixed validation recordings.
    """
    from jiwer import process_words

    units = units_of("mixed-val")
    rows, _ = hypotheses("mixed-val", system)
    wrong = {"kk": 0, "ru": 0, "en": 0}
    words = {"kk": 0, "ru": 0, "en": 0}
    for unit in units.itertuples():
        reference, owner = [], []
        for language, text in zip(unit.languages.split("+"), unit.reference.split(SEP)):
            for word in normalize(text).split():
                reference.append(word)
                owner.append(language)
        for language in owner:
            words[language] += 1
        hypothesis = normalize(rows[unit.unit_id].get("auto", ""))
        if not reference or not hypothesis:
            continue
        out = process_words(" ".join(reference), hypothesis)
        answer = hypothesis.split()
        for chunk in out.alignments[0]:
            if chunk.type != "substitute":
                continue
            for r, h in zip(range(chunk.ref_start_idx, chunk.ref_end_idx), range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                expected_latin = owner[r] == "en"
                got = script_of(answer[h])
                if (expected_latin and got in ("kk", "cyr")) or (not expected_latin and got == "lat" and script_of(reference[r]) != "lat"):
                    wrong[owner[r]] += 1
    return {language: wrong[language] / max(words[language], 1) for language in wrong}


def ksc2_by_source(system):
    """Forced-language WER on KSC2 validation per source (training saw only the crowdsourced source)."""
    from .corpora import load_manifest

    frame = per_unit("ksc2-val", system, "forced")
    domain = dict(zip(*(lambda t: (t.utt_id, t.domain))(load_manifest())))
    frame["source"] = frame.unit_id.map(domain)
    return {source: {**pooled(group.to_dict("records")), "units": int(len(group))} for source, group in frame.groupby("source")}


def prompt_effect(system):
    """Mixed recordings: how often the detected language is the first part's, and WER in both cases."""
    frame = per_unit("mixed-val", system)
    frame["first"] = frame.languages.str.split("+").str[0]
    hit = frame.detected == frame["first"]
    return {"first_language_detected": float(hit.mean()),
            "wer_when_first": pooled(frame[hit].to_dict("records"))["wer"] if hit.any() else None,
            "wer_when_other": pooled(frame[~hit].to_dict("records"))["wer"] if (~hit).any() else None}


def worst(set_name, system, count=8, kind="auto"):
    frame = per_unit(set_name, system, kind).sort_values("errors", ascending=False).head(count)
    return pd.DataFrame({"unit": frame.unit_id, "languages": frame.languages, "errors": frame.errors,
                         "words": frame.reference_words, "reference": frame.reference.str.replace(SEP, " | ").str[:160],
                         "hypothesis": frame.hypothesis.str[:160]})
