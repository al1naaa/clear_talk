"""Validation scores of every measured system, the promotion rules and the app's routes.

Everything here reads the transcripts saved by cleartalk_ml.evaluate; nothing is transcribed again and
no test data are involved. The two rules are the ones pre-registered in the problem contract
(ml_pipeline/PIPELINE.md, step 3):

1. Our model replaces Whisper small as the app's small recognizer if its validation WER on mixed
   recordings is at least 10% lower (relative) with the 95% bootstrap interval of the difference below
   zero, and no language's FLEURS WER is more than 0.5 points worse.
2. Each language is routed to the installed recognizer with the lowest pooled validation WER over that
   language's sets.
"""

import numpy as np

from .evaluate import SETS, bootstrap_difference, details, hyp_path, mixed_by_language, read_hypotheses, units_of
from .paths import LANGUAGES, RESULTS
from .text import pooled

RELATIVE_GAIN = 0.10
LANGUAGE_TOLERANCE = 0.005
LANGUAGE_SETS = {"kk": ("fleurs-val", "ksc2-val"), "ru": ("fleurs-val", "golos-val"), "en": ("fleurs-val",)}
PREFERRED_BACKENDS = ("cuda-int8_float32", "cpu-int8")


def hypotheses(set_name, system, backend=None):
    """(rows, backend) of a system on a set; the graphics-card file is preferred when both exist."""
    name = system.replace("/", "_").replace(":", "-")
    for candidate in ([backend] if backend else PREFERRED_BACKENDS):
        path = RESULTS / "hyp" / f"{set_name}__{name}__{candidate}.jsonl"
        if path.is_file():
            return read_hypotheses(path), candidate
    return {}, None


def single_scores(system, role_units=None):
    """Per set and language: forced and automatic WER/CER of single-language utterances."""
    out = {}
    for set_name in ("fleurs-val", "golos-val", "ksc2-val"):
        units = units_of(set_name) if role_units is None else role_units[set_name]
        rows, backend = hypotheses(set_name, system)
        for language, group in units.groupby("language"):
            done = [u for u in group.unit_id if u in rows]
            if len(done) < len(group):
                continue  # only complete sets are reported
            block = {"units": len(group), "backend": backend}
            for kind in ("forced", "auto"):
                block[kind] = pooled(details(group, rows, kind).to_dict("records"))
            block["language_accuracy"] = float(np.mean([rows[u].get("detected") == language for u in group.unit_id]))
            out[f"{set_name.split('-')[0]}-{language}"] = block
    return out


def mixed_scores(system):
    """WER on mixed validation recordings (automatic language), with errors per language."""
    units = units_of("mixed-val")
    rows, backend = hypotheses("mixed-val", system)
    if any(u not in rows for u in units.unit_id):
        return None
    frame = details(units, rows, "auto")
    return {"units": len(units), "backend": backend, **pooled(frame.to_dict("records")),
            "by_language": mixed_by_language(units, rows), "details": frame}


def promotion(candidate, baseline="small"):
    """Rule 1 of the contract, evaluated on validation only."""
    ours, base = mixed_scores(candidate), mixed_scores(baseline)
    if ours is None or base is None:
        raise ValueError("Mixed validation recordings are not fully transcribed by both systems")
    interval = bootstrap_difference(base["details"], ours["details"])
    relative = (ours["wer"] - base["wer"]) / base["wer"]
    ours_single, base_single = single_scores(candidate), single_scores(baseline)
    languages = {}
    for language in LANGUAGES:
        a, b = base_single.get(f"fleurs-{language}"), ours_single.get(f"fleurs-{language}")
        if a is None or b is None:
            raise ValueError(f"FLEURS validation ({language}) is not fully transcribed by both systems")
        languages[language] = {"baseline_wer": a["forced"]["wer"], "candidate_wer": b["forced"]["wer"],
                               "worse_by": b["forced"]["wer"] - a["forced"]["wer"],
                               "ok": b["forced"]["wer"] - a["forced"]["wer"] <= LANGUAGE_TOLERANCE}
    rule_a = relative <= -RELATIVE_GAIN and interval["high"] < 0
    rule_b = all(item["ok"] for item in languages.values())
    return {"candidate": candidate, "baseline": baseline, "mixed": {"baseline_wer": base["wer"], "candidate_wer": ours["wer"],
            "relative_change": relative, "difference": interval["difference"], "low": interval["low"], "high": interval["high"],
            "recordings": interval["units"]}, "languages": languages, "rule_a_mixed_gain": bool(rule_a),
            "rule_b_no_language_worse": bool(rule_b), "promoted": bool(rule_a and rule_b)}


def routes(systems, covers=None):
    """Rule 2: per language, the system with the lowest pooled validation WER over the language's sets.

    ``systems`` maps a recognizer key of the app profile to the system name used in the transcript files;
    ``covers`` maps a key to the languages the profile lets it serve (the Kazakh fine-tune: Kazakh only).
    A recognizer takes part for a language only if all of that language's sets are transcribed by it.
    """
    table, chosen = {}, {}
    for language, sets in LANGUAGE_SETS.items():
        scores = {}
        for key, system in systems.items():
            if covers is not None and language not in covers.get(key, ()):
                continue
            rows = []
            for set_name in sets:
                units = units_of(set_name, language=language)
                hyp, _ = hypotheses(set_name, system)
                if any(u not in hyp for u in units.unit_id):
                    rows = None
                    break
                rows += details(units, hyp, "forced").to_dict("records")
            if rows:
                scores[key] = pooled(rows)["wer"]
        table[language] = scores
        if scores:
            chosen[language] = min(scores, key=lambda key: (round(scores[key], 5), key))
    return {"routes": chosen, "pooled_validation_wer": table}


def all_sets():
    return sorted(SETS)


__all__ = ["hypotheses", "single_scores", "mixed_scores", "promotion", "routes", "hyp_path"]
