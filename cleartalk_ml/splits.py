"""Cleaning rules, roles (train / val / test), evaluation sets and the frozen single-use test set.

Roles come from the corpora's official parts, never from a random split of utterances:

* FLEURS  - official train / dev / test (disjoint sentences, shared across the three languages).
* Golos   - official train / validation / test; fixed seeded samples of validation and test are scored.
* KSC2    - only the beginning of the archive is on disk. Its official Test folders are complete; its
            Train folders are partial. When the official Dev folders are missing, the official Test
            utterances are divided once, by a seeded draw inside every source, into a validation half
            and a test half: both are outside training, and the test half is never used for choices.

No corpus publishes speaker ids, so "different speakers in every role" cannot be verified here; what
is verified is that no utterance and no FLEURS sentence appears in two roles.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from .paths import LANGUAGES, PIPELINE, PLANS
from .text import normalize

SEED = 20261013
# Scored utterances (fixed seeded samples; FLEURS test: per language). The test samples are smaller than
# the corpora's full test parts because every system is scored through the app's decoding on one laptop
# GPU: 300 per set keeps the one-time final test to about two hours. Decided before any test audio was
# transcribed. A smaller sample is the beginning of the same seeded draw as a larger one.
EVAL_SAMPLE = {"fleurs": {"test": 300}, "golos": {"val": 400, "test": 300}, "ksc2": {"val": 400, "test": 300}}
EVAL_MIXED = {"val": 300, "test": 300}             # mixed recordings scored (the first recipes of each plan)
# Validation / test utterances that were offered to the mixed-recording builder when the recipes were frozen.
MIX_EVAL_POOL = {"golos": {"val": 400, "test": 600}, "ksc2": {"val": 400, "test": 600}}
# One training example costs the same GPU time whether its 30 s window holds one utterance or three
# (0.9 s on the GTX 1660 Ti), so most of an epoch is mixed recordings - they carry about 2.5
# utterances each - and single utterances are seeded samples.
TRAIN_SAMPLE = {"fleurs": 3600, "golos": 800, "ksc2": 1000}   # single utterances per epoch
TRAIN_MIXED = 9000                                             # mixed recordings per epoch
MIX_POOL_SAMPLE = {"ksc2": 6000}                    # utterances offered to the mixed-recording builder
TRAIN_SECONDS = (1.0, 30.0)                         # Whisper sees one 30 s window
# Characters per second. More than 28 means the audio is cut short of its text; less than 1 is a
# fragment of a word in seconds of silence. Short commands with pauses (2-3 characters per second)
# are ordinary Golos speech and stay.
RATE_LIMITS = (1.0, 28.0)
# What one training epoch is made of. Rounds 1 and 2 trained on "base". Round 3 adds the broadcast sources of the
# KSC2 training part (parliament, podcasts, radio, talk shows, TV news), streamed on 2026-10-14; "base" leaves them
# out, so every earlier run draws exactly the utterances it drew before. Chosen with CLEARTALK_DATA.
KSC2_READ = ("crowdsourced",)
COMPOSITIONS = {
    "base": {"single": TRAIN_SAMPLE, "mixed": TRAIN_MIXED, "mix_pool": MIX_POOL_SAMPLE, "plan": "train", "ksc2_broadcast": None},
    # The same epoch plus Kazakh broadcast speech: 500 single utterances per broadcast source, and every kept
    # broadcast utterance offered to the mixed-recording builder next to the 6,000 crowdsourced ones.
    "kazakh-broadcast": {"single": TRAIN_SAMPLE, "mixed": TRAIN_MIXED, "mix_pool": MIX_POOL_SAMPLE, "plan": "train3",
                         "ksc2_broadcast": {"single_per_source": 500, "mix_pool": None}},
}


def composition_name() -> str:
    name = os.environ.get("CLEARTALK_DATA", "base")
    if name not in COMPOSITIONS:
        raise ValueError(f"Unknown training composition {name!r}")
    return name


def composition() -> dict:
    return COMPOSITIONS[composition_name()]
PLAIN = ("golos", "ksc2")                           # corpora with lower-case, unpunctuated transcripts
SEP = " ||| "


def sentence_form(text: str) -> str:
    """Plain transcripts get sentence form: a capital first letter and a full stop. Nothing else is invented."""
    text = " ".join(str(text).split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!…" else text + "."


def with_roles(table: pd.DataFrame) -> pd.DataFrame:
    """Add the column ``role``; only KSC2 can differ from the official part (see the module text)."""
    table = table.copy()
    table["role"] = table["split"]
    ksc2 = table.corpus == "ksc2"
    if ksc2.any() and not ((table.split == "val") & ksc2).any():
        rng = np.random.default_rng(SEED)
        official_test = table[ksc2 & (table.split == "test")].sort_values("utt_id")
        for _, group in official_test.groupby("domain"):
            order = rng.permutation(len(group))
            table.loc[group.index[order[: len(group) // 2]], "role"] = "val"
    return table


def cleaned(table: pd.DataFrame) -> pd.DataFrame:
    """Add ``role``, ``keep``, ``reason`` and the training target ``text``.

    Rules use only an utterance's own audio length and text, or - for the leakage rule - the fact
    that the same sentence is held out; no statistic is estimated from validation or test rows.
    """
    table = with_roles(table)
    key = table.text_raw.map(normalize)
    rate = table.text_raw.str.len() / table.duration.clip(lower=0.1)
    held_out = {(language, text) for language, text in zip(table.language[table.role != "train"], key[table.role != "train"])
                if len(text.split()) >= 4}
    reason = pd.Series("", index=table.index)

    def mark(mask, label):
        reason[mask & (reason == "")] = label

    demo = PLANS / "demo-utterances.json"
    if demo.is_file():  # the packaged demo stays an example the model was never trained on
        used = {item["utt_id"] for item in json.loads(demo.read_text(encoding="utf-8"))["utterances"]}
        mark(table.utt_id.isin(used), "used in the packaged demo")
    mark(~table.exists.astype(str).str.lower().eq("true"), "audio file missing")
    mark(key == "", "empty transcript")
    train = table.role == "train"
    mark(train & (table.duration < TRAIN_SECONDS[0]), "shorter than 1 s")
    mark(train & (table.duration > TRAIN_SECONDS[1]), "longer than 30 s")
    mark(train & ((rate < RATE_LIMITS[0]) | (rate > RATE_LIMITS[1])), "audio and text do not match in length")
    mark(train & pd.Series([(language, text) in held_out for language, text in zip(table.language, key)], index=table.index),
         "same sentence is held out for validation or test")
    table["reason"] = reason
    table["keep"] = reason == ""
    table["text"] = [sentence_form(text) if corpus in PLAIN else " ".join(str(text).split())
                     for text, corpus in zip(table.text_raw, table.corpus)]
    return table


def _sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    frame = frame.sort_values("utt_id")
    if len(frame) <= count:
        return frame
    return frame.iloc[np.sort(np.random.default_rng(seed).permutation(len(frame))[:count])]


def role_table(role: str, purpose: str = "mix") -> pd.DataFrame:
    """Kept single-language utterances of one role.

    purpose="mix"    - the pool for mixed recordings (KSC2 training speech is sampled down);
    purpose="single" - what is scored (val / test) or trained on (train) as single utterances.
    """
    from .corpora import load_manifest

    table = cleaned(load_manifest())
    table = table[(table.role == role) & table.keep]
    frames = []
    recipe = composition()
    for corpus, frame in table.groupby("corpus"):
        if role == "train" and corpus == "ksc2":
            broadcast = frame[~frame.domain.isin(KSC2_READ)]
            frame = frame[frame.domain.isin(KSC2_READ)]
            extra = recipe["ksc2_broadcast"]
            if extra:  # round 3: broadcast speech, sampled per source so no source dominates
                if purpose == "single":
                    frames += [_sample(group, extra["single_per_source"], SEED + 100 + index)
                               for index, (_, group) in enumerate(sorted(broadcast.groupby("domain"), key=lambda item: item[0]))]
                else:
                    frames.append(_sample(broadcast, extra["mix_pool"], SEED + 200) if extra["mix_pool"] else broadcast)
        if role == "train":
            limit = (recipe["single"] if purpose == "single" else recipe["mix_pool"]).get(corpus)
        else:
            limit = (EVAL_SAMPLE if purpose == "single" else MIX_EVAL_POOL).get(corpus, {}).get(role)
            if limit and corpus == "fleurs":  # the same number of utterances for every language
                frames += [_sample(group, limit, SEED + len(corpus) + index)
                           for index, (_, group) in enumerate(sorted(frame.groupby("language"), key=lambda item: item[0]))]
                continue
        frames.append(_sample(frame, limit, SEED + len(corpus)) if limit else frame)
    return pd.concat(frames).sort_values("utt_id").reset_index(drop=True)


def single_example(row) -> dict:
    return {"id": row.utt_id, "kind": "single", "corpus": row.corpus, "language": row.language, "path": row.path,
            "duration": float(row.duration), "text": row.text}


def mixed_example(recipe, texts) -> dict:
    """A training/evaluation example from a recipe; ``texts`` maps utt_id to the target text."""
    return {"id": recipe["mix_id"], "kind": "mixed", "corpus": "mixed", "language": recipe["languages"],
            "duration": float(recipe["duration"]),
            "parts": [{**part, "text": texts.get(part["utt_id"], part["text"])} for part in recipe["parts"]]}


def training_examples(mixed_count=None) -> list[dict]:
    """One epoch: single utterances of the three languages plus mixed-language recordings."""
    from .corpora import load_manifest
    from .mixed import load_plan

    singles = role_table("train", "single")
    texts = dict(zip(*(lambda t: (t.utt_id, t.text))(cleaned(load_manifest()))))
    examples = [single_example(row) for row in singles.itertuples()]
    recipe = composition()
    recipes = load_plan(recipe["plan"])
    examples += [mixed_example(item, texts) for item in recipes[: mixed_count or recipe["mixed"]]]
    return examples


def eval_units(role: str) -> pd.DataFrame:
    """Scoring units of a role: every single utterance and every mixed recording.

    Columns: unit_id, set (e.g. "fleurs-kk", "golos-ru", "mixed"), kind, language, path, duration,
    reference (segments joined by " ||| " in spoken order), languages (same order, "+"-joined).
    """
    from .mixed import plan_table

    if role not in ("val", "test"):
        raise ValueError("Evaluation units exist for val and test")
    singles = role_table(role, "single")
    rows = [{"unit_id": row.utt_id, "set": f"{row.corpus}-{row.language}", "kind": "single", "language": row.language,
             "path": row.path, "duration": round(float(row.duration), 3), "reference": row.text_raw,
             "languages": row.language} for row in singles.itertuples()]
    for row in plan_table(role).head(EVAL_MIXED[role]).itertuples():
        rows.append({"unit_id": row.mix_id, "set": "mixed", "kind": "mixed", "language": "mixed", "path": row.path,
                     "duration": round(float(row.duration), 3),
                     "reference": SEP.join(text for _, text in row.segments), "languages": row.languages})
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


def check_disjoint() -> dict:
    """No utterance and no FLEURS sentence may serve two roles (single or inside a mixed recording)."""
    from .corpora import load_manifest
    from .mixed import load_plan, plan_path

    table = cleaned(load_manifest())
    role_of = dict(zip(table.utt_id, table.role))
    group_of = dict(zip(table.utt_id, table.group))
    used = {role: set(role_table(role, "single").utt_id) | set(role_table(role, "mix").utt_id) for role in ("train", "val", "test")}
    for role in used:
        if plan_path(role).is_file():
            inside = {part["utt_id"] for recipe in load_plan(role) for part in recipe["parts"]}
            wrong = {utt for utt in inside if role_of[utt] != role}
            if wrong:
                raise ValueError(f"{len(wrong)} utterances of another role inside mixed-{role}")
            used[role] |= inside
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    overlap = {f"utterances {a}&{b}": len(used[a] & used[b]) for a, b in pairs}
    sentences = {role: {group_of[utt] for utt in used[role] if group_of[utt].startswith("fleurs-")} for role in used}
    overlap.update({f"FLEURS sentences {a}&{b}": len(sentences[a] & sentences[b]) for a, b in pairs})
    if any(overlap.values()):
        raise ValueError(f"Leakage between roles: {overlap}")
    return overlap


def freeze_test(units: pd.DataFrame) -> dict:
    """Record the test fingerprint the way guard.split() does, for these predefined roles."""
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    import guard

    state = PIPELINE / guard.STATE_FILE
    fingerprint = guard._fingerprint(units)
    if state.exists():
        current = json.loads(state.read_text(encoding="utf-8"))
        if current["test_fingerprint"] != fingerprint:
            raise ValueError("A different test set is already frozen")
        return current
    record = {"test_fingerprint": fingerprint, "n_test": int(len(units)), "test_touches": 0, "duplicates_dropped": 0,
              "split": "official corpus parts (see cleartalk_ml/splits.py)", "created": pd.Timestamp.today().date().isoformat()}
    state.write_text(json.dumps(record, indent=2), encoding="utf-8")
    units.to_csv(PLANS / "test-units.csv", index=False, encoding="utf-8")
    return record


def unit_wer(reference, hypothesis) -> float:
    """Pooled WER over units; references and hypotheses are strings (segments joined by ' ||| ')."""
    from .text import counts, pooled

    return pooled([counts(str(r).replace(SEP, " "), str(h)) for r, h in zip(reference, hypothesis)])["wer"]


def language_list(languages: str) -> list[str]:
    return [language for language in str(languages).split("+") if language in LANGUAGES]
