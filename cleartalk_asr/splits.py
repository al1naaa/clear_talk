"""Train / validation / test frames and the frozen, single-use test set.

The roles come from the official LibriSpeech subsets (speaker-disjoint by construction),
not from a random split. The scoring unit of an evaluation frame is one mixture (two
separated tracks, two references) or one single-speaker utterance.
"""

import json
import sys

import pandas as pd

from .evaluate import eval_tracks
from .paths import PIPELINE

SEP = " ||| "
VAL_SETS = ("val", "v4val")
TEST_SETS = ("test", "v4test")


def units(set_name):
    tracks = eval_tracks(set_name)
    rows = []
    for recipe_id, group in tracks.groupby("recipe_id", sort=True):
        rows.append({"unit_id": f"{set_name}:{recipe_id}", "set": set_name, "kind": group["kind"].iloc[0],
                     "paths": SEP.join(group.path), "speakers": SEP.join(group.speaker.astype(str)),
                     "reference": SEP.join(group.text_ls)})
    return pd.DataFrame(rows)


def eval_frame(sets):
    return pd.concat([units(s) for s in sets], ignore_index=True)


def check_disjoint(train_tracks, val_units, test_units):
    """No speaker and no utterance may appear in two roles."""
    def speakers(frame):
        return {s for joined in frame.speakers for s in joined.split(SEP)}

    train_speakers = set(train_tracks.speaker.astype(str))
    val_speakers, test_speakers = speakers(val_units), speakers(test_units)
    overlap = {"train&val": len(train_speakers & val_speakers), "train&test": len(train_speakers & test_speakers),
               "val&test": len(val_speakers & test_speakers)}
    if any(overlap.values()):
        raise ValueError(f"Speaker leakage between roles: {overlap}")
    return overlap


def freeze_test(test_units):
    """Record the test fingerprint the way guard.split() does, for a predefined split."""
    sys.path.insert(0, str(PIPELINE))
    import guard

    state = PIPELINE / guard.STATE_FILE
    fingerprint = guard._fingerprint(test_units)
    if state.exists():
        current = json.loads(state.read_text())
        if current["test_fingerprint"] != fingerprint:
            raise ValueError("A different test set is already frozen")
        return current
    payload = {"test_fingerprint": fingerprint, "n_test": int(len(test_units)), "test_touches": 0,
               "duplicates_dropped": 0, "split": "predefined: official speaker-disjoint LibriSpeech subsets",
               "created": pd.Timestamp.today().date().isoformat()}
    state.write_text(json.dumps(payload, indent=2))
    return payload


def unit_wer(references, hypotheses, kind="whisper"):
    """metric_fn for guard.final_test: pooled WER over mixtures (best assignment) and singles."""
    from .text import add_counts, edit_counts, pair_counts

    rows = []
    for reference, hypothesis in zip(references, hypotheses):
        refs, hyps = reference.split(SEP), hypothesis.split(SEP)
        rows.append(pair_counts(refs, hyps, kind) if len(refs) > 1 else edit_counts(refs[0], hyps[0], kind))
    return add_counts(rows)["wer"]
