"""Pick the checkpoint to ship from app-path VALIDATION results and export it for the app.

    .venv-train/bin/python scripts/select_checkpoint.py            # choose, export, write the model card
    .venv-train/bin/python scripts/select_checkpoint.py --check    # print the comparison only

Every saved checkpoint of the selected run that has a validation result
(``data/asr/results/val-<run>-step*-app.json``, written by ``cleartalk_asr.evaluate``) is a candidate.
The winner has the lowest WER on the 300 validation mixtures among checkpoints whose clean
single-speaker WER is within 0.5 points of the pretrained model. Test data are not involved.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cleartalk_asr.evaluate import RESULTS  # noqa: E402
from cleartalk_asr.export import export  # noqa: E402
from cleartalk_asr.paths import EXPORT, RUNS  # noqa: E402

CLEAN_TOLERANCE = 0.005


def candidates(run):
    baseline = json.loads((RESULTS / "val-baseline-app.json").read_text())["metrics"]["whisper"]
    rows = []
    for path in sorted(RESULTS.glob(f"val-{run}-step*-app.json")):
        metrics = json.loads(path.read_text())["metrics"]["whisper"]
        step = int(re.search(r"step(\d+)", path.name).group(1))
        rows.append({"step": step, "mixtures_wer": metrics["pairs"]["wer"], "clean_wer": metrics["singles"]["wer"],
                     "eligible": metrics["singles"]["wer"] <= baseline["singles"]["wer"] + CLEAN_TOLERANCE,
                     "result": path.name})
    return baseline, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    selected_path = RUNS / "selected.json"
    selected = json.loads(selected_path.read_text())
    run = selected["run"]
    baseline, rows = candidates(run)
    print(f"pretrained: mixtures {100 * baseline['pairs']['wer']:.2f}%  clean {100 * baseline['singles']['wer']:.2f}%")
    for row in rows:
        print(f"step {row['step']:5d}: mixtures {100 * row['mixtures_wer']:.2f}%  clean {100 * row['clean_wer']:.2f}%"
              f"{'' if row['eligible'] else '  (clean-speech guard failed)'}")
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise SystemExit("No checkpoint passes the clean-speech guard; keep the pretrained recognizer.")
    best = min(eligible, key=lambda row: row["mixtures_wer"])
    print(f"selected: step {best['step']}")
    if args.check:
        return
    config = json.loads((RUNS / run / "config.json").read_text())
    card = {
        "model_name": "ClearTalk Whisper small.en (LoRA fine-tuned)",
        "method": f"LoRA rank {config['rank']} (alpha {2 * config['rank']}, dropout {config['lora_dropout']}) on all attention "
                  "and feed-forward projections; merged into the base weights for inference",
        "trainable_parameters": 6488064, "total_parameters": 248222208,
        "training": {"tracks": config["train_tracks"], "speakers": config["train_speakers"], "mixtures": 8000,
                     "single_speaker_utterances": 3000, "learning_rate": config["lr"],
                     "effective_batch": config["micro_batch"] * config["accumulate"],
                     "steps_per_epoch": config["steps_per_epoch"], "selected_step": best["step"],
                     "precision": "float32", "seed": config["seed"]},
        "selection": {"rule": "lowest validation-mixture WER through the app decoding path, clean speech within 0.5 points",
                      "candidates": rows},
        "data": "LibriSpeech train-clean-100 (CC BY 4.0) with LibriSpeech-PC transcripts, ESC-10 noise; "
                "separated by speechbrain/sepformer-libri2mix",
        "limitations": "Synthetic mixtures of read English audiobooks; 8 kHz separated tracks; not validated on real meetings.",
    }
    export(Path("models/asr-runs") / run / f"step-{best['step']:05d}", EXPORT, card)
    shutil.copy(RESULTS / best["result"], RESULTS / "val-finetuned-app.json")
    selected.update(step=best["step"], validation_candidates=rows)
    selected_path.write_text(json.dumps(selected, indent=2) + "\n")
    print(f"exported step {best['step']} to {EXPORT}")


if __name__ == "__main__":
    main()
