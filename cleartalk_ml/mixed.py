"""Mixed-language recordings built from single-language utterances.

    python -m cleartalk_ml.mixed plan                 # freeze the recipes for train / val / test
    python -m cleartalk_ml.mixed render --split val   # write the validation recordings as WAV files

No open corpus of real Kazakh-Russian-English code-switching was found, so a recording that
changes language is imitated: two to four utterances in different languages are placed one after
another with short pauses and similar loudness. What this does and does not imitate:

* the language changes inside one recording, at a sentence boundary - yes;
* the same person switching language in the middle of a sentence - no: every language change
  here is also a change of speaker and microphone, because each corpus is monolingual.

A recipe is a seeded, frozen list of parts (which utterance, where it starts, how loud). Each role
(train / val / test) draws only from utterances of the same role, and one recording never holds
two versions of the same FLEURS sentence. Recipes are small text files kept with the code, so the
audio can be rebuilt byte for byte on another computer.
"""

import argparse
import json

import numpy as np
import pandas as pd

from .paths import AUDIO, LANGUAGES, PLANS, ROOT, SAMPLE_RATE, rel

MAX_SECONDS = 29.0          # Whisper's window is 30 s
PART_SECONDS = (1.0, 20.0)  # utterances eligible for mixing
GAP_SECONDS = (0.15, 0.7)   # pause between utterances
LEVEL_DB = (-26.0, -20.0)   # RMS level of each part, dBFS
PART_COUNT = {2: 0.45, 3: 0.40, 4: 0.15}
LOOKAHEAD = 12              # candidates tried for the next part before the recording is closed
SEEDS = {"train": 2026101301, "val": 2026101302, "test": 2026101303}
SIZES = {"train": 10000, "val": 300, "test": 400}


def plan_path(split):
    return PLANS / f"mixed-{split}.jsonl"


def make_plan(pool: pd.DataFrame, count: int, seed: int, prefix: str) -> list[dict]:
    """Seeded recipes from ``pool`` (columns: utt_id, language, group, path, duration, text_raw)."""
    rng = np.random.default_rng(seed)
    pool = pool[(pool.duration >= PART_SECONDS[0]) & (pool.duration <= PART_SECONDS[1])]
    by_language = {language: pool[pool.language == language].sort_values("utt_id").reset_index(drop=True)
                   for language in LANGUAGES}
    if any(len(frame) == 0 for frame in by_language.values()):
        raise ValueError("Every language needs utterances in the pool")
    order = {language: list(rng.permutation(len(frame))) for language, frame in by_language.items()}
    durations = {language: frame.duration.to_numpy() for language, frame in by_language.items()}
    group_of = {language: frame.group.to_numpy() for language, frame in by_language.items()}

    def draw(language, budget, used_groups):
        """Next unused utterance of ``language`` that fits; the queue refills when it runs out.

        Only the next LOOKAHEAD candidates are tried: when none of them fits the remaining time,
        the recording simply ends here (searching the whole queue would favour the shortest
        utterances and make long recordings unnaturally dense).
        """
        frame, queue = by_language[language], order[language]
        if len(queue) < LOOKAHEAD:
            queue.extend(rng.permutation(len(frame)))
        for position in range(LOOKAHEAD):
            index = queue[position]
            if durations[language][index] <= budget and group_of[language][index] not in used_groups:
                queue.pop(position)
                return frame.iloc[index]
        return None

    recipes = []
    while len(recipes) < count:
        wanted = int(rng.choice(list(PART_COUNT), p=list(PART_COUNT.values())))
        language = str(rng.choice(LANGUAGES))
        parts, cursor, groups = [], 0.0, set()
        for _ in range(wanted):
            gap = float(rng.uniform(*GAP_SECONDS)) if parts else 0.0
            row = draw(language, MAX_SECONDS - cursor - gap, groups)
            if row is None:
                break
            start = cursor + gap
            parts.append({"utt_id": row.utt_id, "language": language, "path": row.path,
                          "start": round(start, 3), "end": round(start + float(row.duration), 3),
                          "level_db": round(float(rng.uniform(*LEVEL_DB)), 2), "text": row.text_raw})
            cursor = start + float(row.duration)
            groups.add(row.group)
            language = str(rng.choice([other for other in LANGUAGES if other != language]))
        if len(parts) < 2:
            continue
        recipes.append({"mix_id": f"{prefix}-{len(recipes):05d}", "duration": round(cursor, 3),
                        "languages": "+".join(part["language"] for part in parts), "parts": parts})
    return recipes


def save_plan(split, recipes):
    path = plan_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for recipe in recipes:
            handle.write(json.dumps(recipe, ensure_ascii=False) + "\n")
    return path


def load_plan(split) -> list[dict]:
    return [json.loads(line) for line in plan_path(split).read_text(encoding="utf-8").splitlines() if line.strip()]


def plan_table(split) -> pd.DataFrame:
    """One row per recording: id, path of the rendered WAV, reference text, languages in spoken order."""
    rows = []
    for recipe in load_plan(split):
        rows.append({"mix_id": recipe["mix_id"], "split": split, "duration": recipe["duration"],
                     "languages": recipe["languages"], "parts": len(recipe["parts"]),
                     "path": rel(AUDIO / "mixed" / split / f"{recipe['mix_id']}.wav"),
                     "text_raw": " ".join(part["text"] for part in recipe["parts"]),
                     "segments": [(part["language"], part["text"]) for part in recipe["parts"]]})
    return pd.DataFrame(rows)


def render(recipe) -> np.ndarray:
    """The recording as mono float32 at 16 kHz, exactly as the recipe says."""
    from .corpora import read_audio

    out = np.zeros(int(round((recipe["duration"] + 0.05) * SAMPLE_RATE)), dtype=np.float32)
    for part in recipe["parts"]:
        audio = read_audio(part["path"]).astype(np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        if rms > 1e-6:
            audio = audio * (10 ** (part["level_db"] / 20) / rms)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.95:
            audio = audio * (0.95 / peak)
        start = int(round(part["start"] * SAMPLE_RATE))
        audio = audio[: max(0, len(out) - start)]
        out[start:start + len(audio)] = audio
    return out


def render_split(split):
    import soundfile as sf

    folder = AUDIO / "mixed" / split
    folder.mkdir(parents=True, exist_ok=True)
    recipes = load_plan(split)
    for index, recipe in enumerate(recipes, 1):
        path = folder / f"{recipe['mix_id']}.wav"
        if not path.is_file():
            temporary = path.with_suffix(".tmp")
            sf.write(temporary, render(recipe), SAMPLE_RATE, subtype="PCM_16", format="WAV")
            temporary.replace(path)
        if index % 100 == 0:
            print(f"  rendered {index}/{len(recipes)}", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "render"))
    parser.add_argument("--split", choices=("train", "val", "test"), action="append")
    args = parser.parse_args()
    if args.command == "plan":
        from .splits import role_table

        for split in args.split or ("train", "val", "test"):
            if plan_path(split).is_file():
                print(f"{plan_path(split).relative_to(ROOT)} already frozen; delete it deliberately to re-plan")
                continue
            recipes = make_plan(role_table(split), SIZES[split], SEEDS[split], f"mixed-{split}")
            print(save_plan(split, recipes).relative_to(ROOT), len(recipes))
    else:
        for split in args.split or ("val",):
            print(render_split(split).relative_to(ROOT))


if __name__ == "__main__":
    main()
