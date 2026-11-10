"""Run the whole training of the multilingual model unattended - and continue it after any interruption.

    python scripts/run_training.py                       # the small model: start, or continue from wherever it stopped
    python scripts/run_training.py --base kazakh-turbo   # the large model (decoder-only LoRA), same stages
    python scripts/run_training.py --data kazakh-broadcast --rate 5e-4
                                                         # round 3: the small model with Kazakh broadcast speech added;
                                                         # the learning rate of round 1 is carried over (no pilots)
    python scripts/run_training.py --base kazakh-turbo --noise --rate 1e-4
                                                         # round 4: the round-2 large model trained further in noise
    python scripts/run_training.py --base kazakh-turbo --noise --encoder 8 --rate 1e-4
                                                         # round 5: the same, plus LoRA on the top 8 encoder layers

Stages (each is skipped when already finished, and resumed from its last saved state otherwise):

1. Learning-rate pilots: identical data, seed and schedule, PILOT_STEPS optimizer steps each.
2. The learning rate with the lower validation WER on mixed recordings is chosen (a tie goes to the
   lower rate) and written to models/multilingual/runs/lr-choice.json. Validation only.
3. The chosen pilot is continued as the main run to the end of the epoch (or to early stopping).

If the laptop slept, lost power or was restarted: run the same command again. At most 25 optimizer
steps (about six minutes) are repeated.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RATES = {"lr1.5e-4": 1.5e-4, "lr5e-4": 5e-4}
PILOT_STEPS = 80
COMMON = ["--pilot", "--warmup", "30", "--save-every", "25"]
BASE = "small"      # set from --base; run folders of other bases carry a prefix
PREFIX = {"small": "", "kazakh-turbo": "large-"}
DATA_PREFIX = {"base": "", "kazakh-broadcast": "kk-"}
NOISE_START = "models/multilingual/runs/large-lr1.5e-4/step-00600"    # the large model chosen in round 2


def history(run_dir):
    path = run_dir / "history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def train(run, rate, *extra):
    from cleartalk_ml.paths import RUNS

    resume = ["--resume"] if (RUNS / run / "last" / "trainer_state.pt").is_file() else []
    if BASE != "small":  # the large model leaves little graphics memory free: smaller evaluation batches (same numbers)
        extra = (*extra, "--eval-batch", "4")
    if not resume and (RUNS / run / "history.jsonl").is_file():
        # Stopped before the first saved state (only the step-0 evaluation was logged): start this run again.
        (RUNS / run / "history.jsonl").unlink()
    command = [sys.executable, "-X", "utf8", "-m", "cleartalk_ml.train", "--run", run, "--lr", str(rate), "--base", BASE,
               *COMMON, *extra, *resume]
    print("+", " ".join(command[3:]), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    import argparse

    from cleartalk_ml.paths import BASES, RUNS

    global BASE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", choices=sorted(BASES), default="small")
    parser.add_argument("--data", choices=sorted(DATA_PREFIX), default="base", help="training composition (cleartalk_ml.splits)")
    parser.add_argument("--rate", type=float, help="use this learning rate without pilots (carried over from an earlier round)")
    parser.add_argument("--noise", action="store_true",
                        help="round 4: continue the round-2 large model with background noise on half of the examples")
    parser.add_argument("--encoder", type=int, default=0, help="round 5: also train LoRA on this many top encoder layers (with --noise)")
    args = parser.parse_args()
    extra = ()
    if args.noise:
        if args.base != "kazakh-turbo" or not args.rate:
            raise SystemExit("--noise continues the large model: use --base kazakh-turbo --rate 1e-4")
        extra = ("--noise-prob", "0.5", "--init-adapter", NOISE_START, "--epochs", "1.0" if args.encoder else "0.5")
        if args.encoder:  # the encoder's activations need memory: smaller micro-batches, the same 16 examples per step
            extra += ("--encoder-layers", str(args.encoder), "--micro-batch", "2", "--accumulate", "8")
    BASE = args.base
    os.environ["CLEARTALK_DATA"] = args.data      # inherited by every training process started below
    prefix = PREFIX[BASE] + DATA_PREFIX[args.data] + ("noise-" if args.noise else "") + (f"enc{args.encoder}-" if args.encoder else "")
    rates = {prefix + run: rate for run, rate in RATES.items()}
    RUNS.mkdir(parents=True, exist_ok=True)
    choice_path = RUNS / f"{prefix}lr-choice.json"
    if args.rate and not choice_path.is_file():
        mantissa, exponent = f"{args.rate:e}".split("e")
        run = f"{prefix}lr{float(mantissa):g}e{int(exponent)}"     # 5e-4 -> kk-lr5e-4, like the run names of round 1
        rates[run] = args.rate
        choice_path.write_text(json.dumps({"chosen": run, "rule": "learning rate carried over from an earlier round (no pilots)",
                                           "lr": args.rate}, indent=2), encoding="utf-8")
    elif args.rate:
        rates[json.loads(choice_path.read_text(encoding="utf-8"))["chosen"]] = args.rate
    if not choice_path.is_file():
        scores = {}
        for run, rate in rates.items():
            done = [r for r in history(RUNS / run) if r.get("type") == "eval" and r["step"] >= PILOT_STEPS]
            if not done:
                train(run, rate, "--stop-at", str(PILOT_STEPS), "--eval-every", str(PILOT_STEPS))
                done = [r for r in history(RUNS / run) if r.get("type") == "eval" and r["step"] >= PILOT_STEPS]
            scores[run] = {"lr": rate, "step": done[0]["step"], "val_mixed_wer": done[0]["val_mixed"]["wer"],
                           "val_fleurs_wer": {k.split("-")[1]: v["wer"] for k, v in done[0].items() if k.startswith("val_fleurs-")},
                           "val_loss": done[0]["val_loss"]}
        best = min(scores, key=lambda run: (round(scores[run]["val_mixed_wer"], 4), rates[run]))
        choice_path.write_text(json.dumps({"chosen": best, "rule": "lowest validation WER on mixed recordings after "
                                           f"{PILOT_STEPS} steps; a tie goes to the lower rate", "pilots": scores}, indent=2),
                               encoding="utf-8")
    chosen = json.loads(choice_path.read_text(encoding="utf-8"))["chosen"]
    print(f"learning rate: {chosen}", flush=True)
    if not (RUNS / chosen / "finished.json").is_file():
        train(chosen, rates[chosen], "--eval-every", "150" if args.noise else "200", *extra)
    print("TRAINING_FINISHED", (RUNS / chosen / "finished.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
