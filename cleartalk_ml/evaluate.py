"""Score recognizers on the validation sets through the app's decoding path - resumable.

    python -m cleartalk_ml.evaluate --system turbo --set fleurs-val
    python -m cleartalk_ml.evaluate --system small --set mixed-val
    python -m cleartalk_ml.evaluate --system pipeline:auto --set mixed-val     # the app's Conversation mode
    python -m cleartalk_ml.evaluate --report                                    # table of what is finished

A *system* is either one recognizer (``turbo``, ``kazakh``, ``small``, or a CTranslate2 folder below
models/multilingual) or the app's Conversation pipeline (``pipeline:auto`` uses the routes of the model
profile, ``pipeline:<recognizer>`` sends every language to one recognizer).

Single-language utterances get two transcripts:

* ``forced`` - the language is given (the user picked it in the app);
* ``auto``   - the language is detected among Kazakh, Russian and English (the app's automatic mode).

A mixed-language recording gets one transcript: a single recognizer detects one language for the whole
window and writes everything in one pass; the pipeline cuts the recording into turns first.

Every finished unit is appended to ``data/ml/results/hyp/<set>__<system>__<device>-<precision>.jsonl``
at once, so an interrupted run continues where it stopped. Scores are computed from these files.
Only validation sets are accepted here; test sets are scored once, in ml_pipeline/04_eval.py.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from .paths import LANGUAGES, MODELS, READY_MODELS, RESULTS
from .splits import SEP, eval_units
from .text import counts, pooled, segment_counts

SETS = {"fleurs-val": "fleurs", "golos-val": "golos", "ksc2-val": "ksc2", "mixed-val": "mixed"}


def model_dir(key):
    """A ready model by its short name, or any CTranslate2 folder below models/multilingual."""
    return MODELS / (READY_MODELS[key]["ct2"] if key in READY_MODELS else key)


def pick_device(device="auto", precision=None):
    """CUDA when the driver and libraries work, otherwise CPU int8 (the app's portable path).

    On CUDA the default is int8 weights with float32 arithmetic: on the GTX 1660 Ti it gave the same
    transcripts as float32, was the fastest of the three and needs 2.3 GB instead of 5.3 GB; float16
    is correct there too but several times slower (this card has no fast half-precision path).
    """
    if device == "auto":
        try:
            import torch  # noqa: F401  (on Windows this makes the bundled CUDA libraries loadable)
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    return device, precision or ("int8_float32" if device == "cuda" else "int8")


def load_model(key, device="auto", precision=None, threads=4):
    from faster_whisper import WhisperModel

    device, precision = pick_device(device, precision)
    model = WhisperModel(str(model_dir(key)), device=device, compute_type=precision, cpu_threads=threads,
                         num_workers=1, local_files_only=True)
    return model, device, precision


def units_of(set_name, role="val", language=None, limit=None):
    """Scoring units of one set (``role`` is "val" here; step 14 passes "test")."""
    frame = eval_units(role)
    frame = frame[frame.set.str.split("-").str[0] == SETS[set_name]]
    if language:
        frame = frame[frame.language == language]
    frame = frame.sort_values("unit_id").reset_index(drop=True)
    if limit:  # a fixed, seeded subset per language (never chosen by results)
        frame = pd.concat([group.sample(min(limit, len(group)), random_state=20261013)
                           for _, group in frame.groupby("language")]).sort_values("unit_id").reset_index(drop=True)
    return frame


def hyp_path(set_name, system, device, precision):
    name = system.replace("/", "_").replace(":", "-")
    return RESULTS / "hyp" / f"{set_name}__{name}__{device}-{precision}.jsonl"


def without_loops(rows):
    """Stored transcripts as the app now writes them: repetition loops cut per turn (backend/loops.py).

    Transcripts made before the loop cut existed are compared with new ones through this; it changes nothing in
    transcripts that already went through it."""
    from backend.loops import cut_loops

    out = {}
    for key, row in rows.items():
        row = dict(row)
        if row.get("turns"):
            row["turns"] = [{**turn, "text": cut_loops(turn["text"])[0]} for turn in row["turns"]]
            row["auto"] = row["forced"] = " ".join(turn["text"] for turn in row["turns"])
        else:
            for kind in ("auto", "forced"):
                if isinstance(row.get(kind), str):
                    row[kind] = cut_loops(row[kind])[0]
        out[key] = row
    return out


def read_hypotheses(path):
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["utt_id"]] = row
    return rows


# ----------------------------------------------------------------------------- transcription
def transcribe_single(model, audio, true_language, forced_only=False):
    """Auto and forced transcripts of one utterance with as few decoding passes as possible.

    ``forced_only`` skips the automatic pass (used for the large reference models in the final test,
    where their language detection was already measured on validation): ``auto`` is then absent.
    """
    from backend.multilingual import decoding_policy, restricted_language, transcribe_window

    if forced_only:
        return {"detected": None, "forced": transcribe_window(model, audio, true_language)["text"]}

    def run(language):
        policy = decoding_policy(language or "en")
        policy["language"] = language
        segments, info = model.transcribe(audio, **policy)
        return " ".join(s.text.strip() for s in segments if s.text.strip()), info

    text, info = run(None)  # Whisper's own detection over all its languages
    picked, share, shares = restricted_language(info.all_language_probs)
    auto = text if info.language == picked else run(picked)[0]
    forced = auto if picked == true_language else (text if info.language == true_language else run(true_language)[0])
    return {"detected": picked, "share": round(share, 4), "shares": {k: round(v, 4) for k, v in shares.items()},
            "unrestricted": info.language, "auto": auto, "forced": forced}


def transcribe_mixed(model, audio):
    """One recognizer, one pass: detect a language for the window, then write the whole window."""
    from backend.multilingual import detect_language, transcribe_window

    found = detect_language(model, audio)
    return {"detected": found["language"], "auto": transcribe_window(model, audio, found["language"])["text"]}


class Systems:
    """Loads what a system needs once and transcribes units of any kind."""

    def __init__(self, system, device="auto", precision=None, threads=4, forced_only=False):
        self.system = system
        self.forced_only = forced_only
        self.device, self.precision = pick_device(device, precision)
        if system.startswith("pipeline:"):
            from backend.config import Settings
            from backend.conversation import ConversationPipeline

            settings = Settings(device=self.device, cpu_threads=threads)
            self.pipeline, self.model_key = ConversationPipeline(settings), system.split(":", 1)[1]
            self.device, self.precision = self.pipeline.device()
        else:
            self.pipeline = None
            self.model = load_model(system, self.device, self.precision, threads)[0]

    def transcribe(self, unit, audio):
        if self.pipeline is not None:
            if self.model_key == "mixed":  # the app's setting "languages change inside turns"
                result = self.pipeline.transcribe(audio, language="mixed", model="auto")
            elif self.model_key == "mixed-fast":  # the same without the language detector
                result = self.pipeline.transcribe(audio, language="mixed", model="auto", skip_detection=True)
            elif self.model_key == "auto-safe":  # per-turn routes, uncertain turns to the recognizer of the mixed setting
                result = self.pipeline.transcribe(audio, language="auto", model="auto", uncertain_to_mixed=True)
            elif self.model_key == "auto-crosscheck":  # the current per-turn pipeline plus the cross-check of English turns
                result = self.pipeline.transcribe(audio, language="auto", model="auto", cross_check=True)
            elif self.model_key == "auto-crosscheck-kk":  # the cross-check, taking over only turns our model heard as Kazakh
                result = self.pipeline.transcribe(audio, language="auto", model="auto", cross_check="kazakh")
            elif self.model_key == "auto-region":  # the current per-turn pipeline, recognising whole stretches of speech
                result = self.pipeline.transcribe(audio, language="auto", model="auto", by_region=True)
            elif self.model_key == "mixed-region":
                result = self.pipeline.transcribe(audio, language="mixed", model="auto", by_region=True)
            elif self.model_key == "auto-overlap":  # region-first, with people talking at once separated
                result = self.pipeline.transcribe(audio, language="auto", model="auto", by_region=True, separate_overlaps=True)
            elif self.model_key == "auto-before":  # the app as round 4 began: none of its protections
                result = self.pipeline.transcribe(audio, language="auto", model="auto", by_region=False, separate_overlaps=False,
                                                  hide_background=False, cross_check=False, cut_repeats=False, text_first=False)
            elif self.model_key.startswith("auto+"):  # the per-turn pipeline with exactly the named options on
                flags = set(self.model_key.split("+")[1:])
                result = self.pipeline.transcribe(audio, language="auto", model="auto", by_region="region" in flags,
                                                  separate_overlaps="overlap" in flags, hide_background="background" in flags,
                                                  cross_check="kazakh" if "kk" in flags else False, text_first="text" in flags)
            elif self.model_key == "auto-validated":  # per-turn routes with both options off, whatever the profile says
                result = self.pipeline.transcribe(audio, language="auto", model="auto", skip_detection=False, uncertain_to_mixed=False, cross_check=False)
            else:
                result = self.pipeline.transcribe(audio, language="auto", model=self.model_key)
            turns = [{"start": t["start"], "end": t["end"], "speaker": t["speaker"], "language": t["language"],
                      "recognizer": t["recognizer"], "text": t["text"], **({"overlap": True} if t.get("overlap") else {})} for t in result["turns"]]
            text = " ".join(turn["text"] for turn in turns)
            return {"auto": text, "forced": text, "turns": turns,
                    "detected": "+".join(dict.fromkeys(turn["language"] for turn in turns))}
        if unit.kind == "mixed":
            return transcribe_mixed(self.model, audio)
        return transcribe_single(self.model, audio, unit.language, self.forced_only)


def run_units(system, units, path, device="auto", precision=None, threads=4, forced_only=False):
    """Transcribe ``units`` with ``system`` into ``path`` (appending; finished units are skipped)."""
    from .awake import keep_awake
    from .corpora import read_audio

    keep_awake()
    done = read_hypotheses(path)
    todo = units[~units.unit_id.isin(done)]
    if not len(todo):
        return path
    engine = Systems(system, device, precision, threads, forced_only)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{system}: {len(units)} units, {len(done)} already done, {len(todo)} to do on {engine.device} "
          f"({engine.precision})", flush=True)
    started, audio_seconds = time.monotonic(), 0.0
    with path.open("a", encoding="utf-8") as out:
        for index, unit in enumerate(todo.itertuples(), 1):
            t0 = time.monotonic()
            result = engine.transcribe(unit, read_audio(unit.path))
            result.update(utt_id=unit.unit_id, language=unit.language, duration=round(unit.duration, 2),
                          seconds=round(time.monotonic() - t0, 2))
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            audio_seconds += unit.duration
            if index % 25 == 0 or index == len(todo):
                elapsed = time.monotonic() - started
                print(f"  {index}/{len(todo)} · {audio_seconds / elapsed:.1f}x real time · "
                      f"about {(len(todo) - index) * elapsed / index / 60:.0f} min left", flush=True)
    return path


# ----------------------------------------------------------------------------- scoring
def details(units, rows, kind="auto"):
    """One row per scored unit: word and character edit counts of the ``kind`` transcript."""
    out = []
    for unit in units.itertuples():
        row = rows.get(unit.unit_id)
        if row is None:
            continue
        hypothesis = row.get(kind, row.get("auto", row.get("forced", "")))
        out.append({"unit_id": unit.unit_id, "set": unit.set, "language": unit.language, "detected": row.get("detected"),
                    "hypothesis": hypothesis, **counts(unit.reference.replace(SEP, " "), hypothesis)})
    return pd.DataFrame(out)


def mixed_by_language(units, rows):
    """Errors inside mixed recordings attributed to the language of the reference words."""
    total = {language: {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0} for language in LANGUAGES}
    for unit in units.itertuples():
        row = rows.get(unit.unit_id)
        if row is None:
            continue
        segments = list(zip(unit.languages.split("+"), unit.reference.split(SEP)))
        for language, block in segment_counts(segments, row.get("auto", "")).items():
            for key, value in block.items():
                total[language][key] += value
    for block in total.values():
        block["wer"] = (block["substitutions"] + block["deletions"] + block["insertions"]) / max(block["reference_words"], 1)
    return total


def bootstrap_difference(details_a, details_b, iterations=2000, seed=0):
    """95% interval of WER(b) - WER(a), resampling units (paired: the same units for both systems)."""
    merged = details_a.merge(details_b, on="unit_id", suffixes=("_a", "_b"))
    errors = {side: (merged[f"substitutions_{side}"] + merged[f"deletions_{side}"] + merged[f"insertions_{side}"]).to_numpy()
              for side in ("a", "b")}
    words = merged.reference_words_a.to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iterations):
        index = rng.integers(0, len(merged), len(merged))
        diffs.append((errors["b"][index].sum() - errors["a"][index].sum()) / max(words[index].sum(), 1))
    return {"difference": float((errors["b"].sum() - errors["a"].sum()) / max(words.sum(), 1)),
            "low": float(np.percentile(diffs, 2.5)), "high": float(np.percentile(diffs, 97.5)), "units": int(len(merged))}


def score_file(path, units):
    """Pooled WER/CER per language (or for the mixed set) of the forced and automatic transcripts."""
    rows = read_hypotheses(path)
    result = {}
    for language, group in units.groupby("language"):
        items = [rows[u] for u in group.unit_id if u in rows]
        if not items:
            continue
        block = {"units": len(items), "expected": len(group)}
        for kind in ("forced", "auto"):
            block[kind] = pooled(details(group, rows, kind).to_dict("records"))
        if language != "mixed":
            block["language_accuracy"] = sum(row.get("detected") == language for row in items) / len(items)
        block["seconds_per_audio_second"] = sum(row["seconds"] for row in items) / max(sum(row["duration"] for row in items), 1e-9)
        result[language] = block
    return result


def report(set_names=None, role="val"):
    lines = []
    for set_name in set_names or sorted(SETS):
        units = units_of(set_name, role)
        for path in sorted((RESULTS / "hyp").glob(f"{set_name}__*.jsonl")):
            _, system, backend = path.stem.split("__")
            for language, block in score_file(path, units).items():
                lines.append({"set": set_name, "system": system, "backend": backend, "language": language,
                              "units": f"{block['units']}/{block['expected']}",
                              "WER forced %": round(100 * block["forced"]["wer"], 2),
                              "CER forced %": round(100 * block["forced"]["cer"], 2),
                              "WER auto %": round(100 * block["auto"]["wer"], 2),
                              "language found %": round(100 * block["language_accuracy"], 1) if "language_accuracy" in block else None,
                              "speed (x real time)": round(1 / max(block["seconds_per_audio_second"], 1e-9), 1)})
    return pd.DataFrame(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", "--model", dest="system",
                        help="turbo, kazakh, small, a CTranslate2 folder, pipeline:auto or pipeline:<recognizer>")
    parser.add_argument("--set", choices=sorted(SETS), action="append")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--precision", help="default: int8 on CPU, int8_float32 on CUDA")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    sets = args.set or sorted(SETS)
    if not args.report:
        if not args.system:
            parser.error("--system is required")
        device, precision = pick_device(args.device, args.precision)
        for set_name in sets:
            if args.system.startswith("pipeline:") and set_name != "mixed-val":
                continue  # for one utterance in one language the pipeline is just its route
            print(f"== {set_name}", flush=True)
            run_units(args.system, units_of(set_name), hyp_path(set_name, args.system, device, precision),
                      device, precision, args.threads)
    print(report(sets).to_string(index=False))


if __name__ == "__main__":
    main()
