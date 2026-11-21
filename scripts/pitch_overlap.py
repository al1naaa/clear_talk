"""Compare two frozen separators on validation; measure cpWER and SI-SDRi on unseen pairs.

Run --split validation --separator libri then wham, --select, then --split test.
Reference waveforms are used exclusively in scoring. Source clips are never published.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.overlap import Separator
from scripts.pitch_scoring import stream_errors, signal_scores
from scripts.external_pilot import noisy
from scripts.train_three_language_asr import split_rows, wave
from scripts.validate_pitch_asr import AuditPipeline, INCUMBENT

PAIRS = (("kk", "ru"), ("ru", "en"), ("en", "kk"))


def mixtures(sets, n):
    # Each speaker clip participates in one pair, shared across systems and noise views.
    positions = defaultdict(int)
    for a, b in PAIRS:
        for j in range(n):
            aid, x, at = sets[a][positions[a]]
            bid, y, bt = sets[b][positions[b]]
            positions[a] += 1
            positions[b] += 1
            x = x * (0.07 / max(float(np.sqrt(np.mean(x * x))), 1e-8))
            y = (
                y
                * (0.07 / max(float(np.sqrt(np.mean(y * y))), 1e-8))
                * 10 ** (((-3, 0, 3)[j % 3]) / 20)
            )
            offset = round(0.25 * min(len(x), len(y)))
            length = max(len(x), len(y) + offset)
            src = np.zeros((2, length), dtype="float32")
            src[0, : len(x)] = x
            src[1, offset : offset + len(y)] = y
            uid = f"{a}+{b}-{j:02d}"
            for condition in ("clean", "white-10"):
                mix = src.sum(axis=0)
                if condition != "clean":
                    mix = noisy(mix, aid + bid, 10)
                yield uid, condition, mix, src, [at, bt]


def summaries(rows):
    out = {}
    groups = defaultdict(list)
    for row in rows:
        for key in (
            "all",
            row["condition"],
            row["pair"],
            row["pair"] + "_" + row["condition"],
        ):
            groups[key].append(row)
    for group, rs in groups.items():
        errors = sum(r["errors"] for r in rs)
        words = sum(r["words"] for r in rs)
        out[group] = dict(
            n=len(rs),
            errors=errors,
            words=words,
            cpwer=errors / words,
            si_sdri_db=float(np.mean([r["si_sdri_db"] for r in rs])),
            two_tracks=sum(r["two_tracks"] for r in rs),
        )
    return out


class Capture(Separator):
    def split(self, audio):
        result = super().split(audio)
        self.estimates = result[0]
        return result


def evaluate(sets, n, sep, folder, path, threads):
    from speechbrain.inference.separation import SepformerSeparation

    torch.set_num_threads(threads)
    pipe = AuditPipeline(folder, threads)
    capture = Capture(pipe.settings, "cpu")
    if sep in ("wham", "adapted"):
        source = ROOT / (
            "models/sepformer"
            if sep == "wham"
            else "models/candidates-v3/sepformer-libri2mix"
        )
        capture.model = SepformerSeparation.from_hparams(
            source=str(source), savedir=str(source), run_opts={"device": "cpu"}
        )
        if sep == "adapted":
            decoder = torch.load(
                ROOT / "models/pitch-separator/selected.ckpt",
                map_location="cpu",
                weights_only=True,
            )
            capture.model.mods.decoder.load_state_dict(decoder)
        capture.model.mods.eval()
    pipe._separator = capture
    path.parent.mkdir(exist_ok=True, parents=True)
    rows = (
        [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    )
    done = {(r["uid"], r["condition"]) for r in rows}
    for uid, condition, mix, src, refs in mixtures(sets, n):
        if (uid, condition) in done:
            continue
        result = pipe.transcribe_two_voices(mix)
        estimates = capture.estimates
        before, after = signal_scores(mix, src, estimates)
        err, words = stream_errors(refs, result["turns"])
        row = dict(
            uid=uid,
            pair=uid.split("-")[0],
            condition=condition,
            errors=err,
            words=words,
            si_sdri_db=after - before,
            two_tracks=bool(result.get("two_voices")),
        )
        rows.append(row)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(uid, condition, err, words, flush=True)
    assert (
        len(rows)
        == len(done | {(r["uid"], r["condition"]) for r in rows})
        == 2 * n * len(PAIRS)
    )
    return summaries(rows)


def score_isolated_controls(sets, holdout):
    """Diagnostic ceiling from measured ASR on the original single-speaker input files."""
    rows = [json.loads(line) for line in holdout.read_text().splitlines()]
    hypotheses = {r["uid"]: r["hypothesis"] for r in rows if r["condition"] == "clean"}
    assert len(hypotheses) == 54
    positions = defaultdict(int)
    result = {}
    for a, b in PAIRS:
        errors = words = 0
        for _ in range(6):
            items = [sets[l][positions[l]] for l in (a, b)]
            positions[a] += 1
            positions[b] += 1
            refs = [item[2] for item in items]
            turns = [
                dict(start=0, speaker=i, text=hypotheses[item[0]])
                for i, item in enumerate(items)
            ]
            err, ref_words = stream_errors(refs, turns)
            errors += err
            words += ref_words
        result[a + "+" + b] = dict(errors=errors, words=words, cpwer=errors / words)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "models/pitch-asr")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--separator", choices=("libri", "wham", "adapted"), default="libri"
    )
    parser.add_argument("--select", action="store_true")
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Score saved original-source transcripts for the same test pairs",
    )
    parser.add_argument("--threads", type=int, default=3)
    opt = parser.parse_args()
    work = opt.work / "overlap"
    result_path = work / "selection.json"
    if opt.isolated:
        from scripts.final_pitch_controls import controls

        step = json.loads((opt.work / "selection.json").read_text())["selected_step"]
        result = score_isolated_controls(controls(), opt.work / f"holdout/{step}.jsonl")
        (work / "isolated-test.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    if opt.select:
        results = {
            s: json.loads((work / f"validation-{s}.json").read_text())
            for s in ("libri", "wham")
        }
        decoder = json.loads(
            (ROOT / "models/pitch-separator/selection.json").read_text()
        )
        if decoder["epoch"]:
            rs = split_rows("val", 6, "separator-validation")
            sets = {
                l: [(r["utt_id"], wave(r), r["text_raw"]) for r in rows]
                for l, rows in rs.items()
            }
            results["adapted"] = evaluate(
                sets,
                3,
                "adapted",
                INCUMBENT,
                work / "validation-adapted.jsonl",
                opt.threads,
            )
            (work / "validation-adapted.json").write_text(
                json.dumps(results["adapted"], indent=2) + "\n"
            )
        old = results["libri"]["all"]
        safe = [
            s for s, m in results.items() if m["all"]["si_sdri_db"] >= old["si_sdri_db"]
        ]
        choice = min(safe, key=lambda s: results[s]["all"]["cpwer"])
        result_path.write_text(
            json.dumps(
                dict(
                    selected=choice,
                    acoustic_decoder_epoch=decoder["epoch"],
                    validation=results,
                    rule="Lowest pooled cpWER with mean SI-SDRi at least that of Libri2Mix; retain Libri2Mix on ties.",
                ),
                indent=2,
            )
            + "\n"
        )
        return
    if opt.split == "validation":
        n = 3
        rs = split_rows("val", 2 * n, "separator-validation")
        sets = {
            l: [(r["utt_id"], wave(r), r["text_raw"]) for r in rows]
            for l, rows in rs.items()
        }
        sep, folder = opt.separator, INCUMBENT
    else:
        from scripts.final_pitch_controls import controls

        sets = controls()
        n = 6
        sep = json.loads(result_path.read_text())["selected"]
        step = json.loads((opt.work / "selection.json").read_text())["selected_step"]
        folder = opt.work / f"ct2-{step:03d}" if step else INCUMBENT
    prefix = f"{opt.split}-{sep}"
    result = evaluate(sets, n, sep, folder, work / (prefix + ".jsonl"), opt.threads)
    (work / (prefix + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    if opt.split == "test" and (sep != "libri" or step):
        baseline = evaluate(
            sets, n, "libri", INCUMBENT, work / "test-baseline.jsonl", opt.threads
        )
        (work / "test-baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")


if __name__ == "__main__":
    main()
