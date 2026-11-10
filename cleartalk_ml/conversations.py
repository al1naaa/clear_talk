"""Check and tune the speaker labels of Conversation mode on assembled validation conversations.

    python -m cleartalk_ml.conversations          # tune on half A, report on half B, write the result

No corpus on disk has speaker ids, so a conversation with KNOWN speakers is assembled like this:
each "speaker" is one FLEURS validation utterance, cut at quiet points into two or three pieces;
the pieces of two to four utterances are interleaved with short pauses. Pieces of one utterance are
certainly the same person; two utterances are certainly different people only if they differ in
language or in the reader's gender, so every conversation uses distinct (language, gender) pairs.

That makes the check easier than real life - speakers of the same gender and language are never
compared - and the result must be read as "the method works on clearly different voices".
Only validation utterances are used; the test part is not touched.
"""

import argparse
import json

import numpy as np

from .paths import LANGUAGES, RESULTS, SAMPLE_RATE, SPEAKER_MODEL

SEED = 2026101305
COUNT = 80                      # conversations: the first half tunes, the second half reports
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
PIECE_SECONDS = 1.5             # shortest piece
GAP_SECONDS = (0.25, 0.8)
FRAME = 0.01


def quiet_cuts(audio, pieces):
    """Sample positions that cut ``audio`` into ``pieces`` parts at its quietest moments."""
    hop = int(0.02 * SAMPLE_RATE)
    energy = np.array([float(np.mean(audio[i:i + hop] ** 2)) for i in range(0, len(audio) - hop, hop)])
    smooth = np.convolve(energy, np.ones(5) / 5, mode="same")
    cuts = []
    for k in range(1, pieces):
        low, high = int(len(smooth) * (k / pieces - 0.12)), int(len(smooth) * (k / pieces + 0.12))
        cuts.append((low + int(np.argmin(smooth[low:high]))) * hop)
    return cuts


def build(count=COUNT, seed=SEED):
    """Recipes: for each conversation the pieces in spoken order as (utt_id, path, from, to, speaker)."""
    from .corpora import load_manifest, read_audio

    table = load_manifest()
    pool = table[(table.corpus == "fleurs") & (table.split == "val") & table.duration.between(6, 15)
                 & table.gender.isin(["male", "female"])].sort_values("utt_id")
    rng = np.random.default_rng(seed)
    voices = [(language, gender) for language in LANGUAGES for gender in ("male", "female")]
    conversations = []
    while len(conversations) < count:
        speakers = int(rng.choice([2, 3, 4]))
        chosen = [voices[i] for i in rng.permutation(len(voices))[:speakers]]
        queue = []
        for index, (language, gender) in enumerate(chosen):
            candidates = pool[(pool.language == language) & (pool.gender == gender)]
            row = candidates.iloc[int(rng.integers(len(candidates)))]
            audio = read_audio(row.path)
            pieces = int(rng.choice([2, 3])) if len(audio) / SAMPLE_RATE >= 3 * PIECE_SECONDS * 1.5 else 2
            edges = [0, *quiet_cuts(audio, pieces), len(audio)]
            spans = [(a, b) for a, b in zip(edges, edges[1:]) if (b - a) / SAMPLE_RATE >= PIECE_SECONDS]
            queue.append([{"utt_id": row.utt_id, "path": row.path, "from": int(a), "to": int(b), "speaker": index,
                           "voice": f"{language}-{gender}"} for a, b in spans])
        order, last = [], None
        while any(queue):
            options = [i for i, items in enumerate(queue) if items and i != last] or [i for i, items in enumerate(queue) if items]
            last = int(rng.choice(options))
            order.append({**queue[last].pop(0), "gap": round(float(rng.uniform(*GAP_SECONDS)), 3)})
        if len({piece["speaker"] for piece in order}) == speakers:
            conversations.append({"id": f"conversation-{len(conversations):03d}", "speakers": speakers, "pieces": order})
    return conversations


def render(conversation):
    """Audio and the reference (start, end, speaker) of every piece."""
    from .corpora import read_audio

    parts, reference, cursor = [], [], 0.0
    for piece in conversation["pieces"]:
        audio = read_audio(piece["path"])[piece["from"]:piece["to"]].astype(np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        audio = np.clip(audio * (10 ** (-23 / 20) / max(rms, 1e-6)), -0.95, 0.95)
        reference.append((cursor, cursor + len(audio) / SAMPLE_RATE, piece["speaker"]))
        gap = np.zeros(int(piece["gap"] * SAMPLE_RATE), dtype=np.float32)
        parts += [audio, gap]
        cursor += (len(audio) + len(gap)) / SAMPLE_RATE
    return np.concatenate(parts), reference


def score(reference, turns, duration):
    """Share of speech time given to the wrong speaker (after the best label matching) and missed speech."""
    from scipy.optimize import linear_sum_assignment

    frames = int(duration / FRAME) + 1
    ref = np.full(frames, -1)
    hyp = np.full(frames, -1)
    for start, end, speaker in reference:
        ref[int(start / FRAME):int(end / FRAME)] = speaker
    for turn in turns:
        hyp[int(turn.start / FRAME):int(turn.end / FRAME)] = turn.speaker
    both = (ref >= 0) & (hyp >= 0)
    table = np.zeros((ref.max() + 1, max(hyp.max(), 0) + 1))
    for r, h in zip(ref[both], hyp[both]):
        table[r, h] += 1
    rows, columns = linear_sum_assignment(-table)
    matched = table[rows, columns].sum()
    return {"confusion": float(1 - matched / max(both.sum(), 1)), "missed": float(((ref >= 0) & (hyp < 0)).mean() / max((ref >= 0).mean(), 1e-9)),
            "found_speakers": int(len(set(hyp[hyp >= 0].tolist()))), "true_speakers": int(len(set(ref[ref >= 0].tolist())))}


def main():
    from backend import speakers

    from .awake import keep_awake

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    keep_awake()
    encoder = speakers.SpeakerEncoder(SPEAKER_MODEL, args.device, 4)
    conversations = build()
    rows = []
    for index, conversation in enumerate(conversations):
        audio, reference = render(conversation)
        regions = speakers.speech_regions(audio)
        windows = speakers.windows_of(regions)
        vectors = encoder.embed(audio, [(start, end) for _, start, end in windows])
        for threshold in THRESHOLDS:
            turns = speakers.turns_from(regions, windows, speakers.cluster(vectors, None, threshold))
            rows.append({"id": conversation["id"], "half": "tune" if index < len(conversations) // 2 else "check",
                         "threshold": threshold, **score(reference, turns, len(audio) / SAMPLE_RATE)})
        turns = speakers.turns_from(regions, windows, speakers.cluster(vectors, conversation["speakers"]))
        rows.append({"id": conversation["id"], "half": "tune" if index < len(conversations) // 2 else "check",
                     "threshold": "count given", **score(reference, turns, len(audio) / SAMPLE_RATE)})
        if (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(conversations)} conversations", flush=True)

    def summary(half, threshold):
        items = [row for row in rows if row["half"] == half and row["threshold"] == threshold]
        # "missed" is not summarized: a reference piece includes the reader's own pauses, which the
        # speech detector rightly leaves out, so it would mostly count silence.
        return {"conversations": len(items), "confusion": float(np.mean([row["confusion"] for row in items])),
                "speaker_count_right": float(np.mean([row["found_speakers"] == row["true_speakers"] for row in items]))}

    tuning = {threshold: summary("tune", threshold) for threshold in THRESHOLDS}
    best = min(THRESHOLDS, key=lambda t: (tuning[t]["confusion"] + (1 - tuning[t]["speaker_count_right"]), abs(t - 0.75)))
    result = {"seed": SEED, "tuning_half": {str(k): v for k, v in tuning.items()}, "chosen_threshold": best,
              "check_half": summary("check", best), "check_half_count_given": summary("check", "count given"),
              "threshold_in_app": speakers.DISTANCE_THRESHOLD,
              "note": "Assembled validation conversations; speakers always differ in language or gender."}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "speaker-labels-validation.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
