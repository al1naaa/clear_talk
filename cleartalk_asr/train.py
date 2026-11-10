"""LoRA fine-tuning of Whisper small.en on ClearTalk's separated tracks (steps 10-11).

    python -m cleartalk_asr.train --run main --lr 5e-4                  # full run with early stopping
    python -m cleartalk_asr.train --run pilot-lr1e-4 --lr 1e-4 --stop-at 120 --eval-every 120 --pilot
    python -m cleartalk_asr.train --run main --resume                    # continue after an interruption
    python -m cleartalk_asr.train --run smoke --smoke                    # tiny end-to-end check

Every ``--eval-every`` optimizer steps the model transcribes (greedy) the validation mixtures,
a fixed subset of clean validation utterances and a fixed subset of *training* tracks, so the
train/validation gap is visible. The adapter with the lowest validation mixture WER whose clean-
speech WER stays within 0.5 points of the pretrained model is kept in ``best/``; training stops
after ``--patience`` evaluations without improvement. Test data are never opened here.
"""

import argparse
import contextlib
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch

from .dataset import TrackDataset, collate
from .evaluate import eval_tracks, score, transcribe_hf
from .mixtures import load_tracks
from .paths import BASE_WHISPER, RUNS

SI_SDR_MIN = -5.0  # step 5 rule: failed separations are not training targets
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
CLEAN_TOLERANCE = 0.005  # selection guard on clean single-speaker WER (absolute)
MIN_IMPROVEMENT = 0.001  # early-stopping threshold on validation WER (absolute)


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def training_tracks(max_pairs=None):
    """Filtered training tracks; ``max_pairs`` fixes a plan prefix (pilots run while the build continues)."""
    tracks = load_tracks("train")
    keep = (tracks.kind == "single") | (tracks.si_sdr_db >= SI_SDR_MIN)
    tracks = tracks[keep]
    if max_pairs is not None:
        index = tracks.recipe_id.str.extract(r"train-pair-(\d+)$")[0].astype(float)
        tracks = tracks[index < max_pairs]
        if len(set(tracks.recipe_id)) < 0.9 * max_pairs:
            raise SystemExit(f"Only {len(set(tracks.recipe_id))} of the first {max_pairs} pairs are built yet")
    return tracks.sort_values("track_id").reset_index(drop=True)


def monitor_sets(train, seed, pilot=False):
    """Fixed evaluation subsets, chosen by seed only (never by results)."""
    rng = np.random.default_rng(seed)
    val = eval_tracks("val")
    pair_ids = sorted(val[val.kind == "pair"].recipe_id.unique())
    clean_ids = sorted(val[val.kind == "single"].recipe_id.unique())
    # Periodic monitoring uses fixed subsets (full validation is scored later through the app path).
    pair_ids = list(rng.choice(pair_ids, 120 if pilot else 200, replace=False))
    clean_pick = list(rng.choice(clean_ids, 40 if pilot else 60, replace=False))
    val_monitor = val[val.recipe_id.isin(set(pair_ids) | set(clean_pick))]
    pairs = train[train.kind == "pair"]
    complete = pairs.groupby("recipe_id").track_id.transform("size") == 2
    train_pairs = sorted(pairs[complete].recipe_id.unique())
    train_singles = sorted(train[train.kind == "single"].recipe_id.unique())
    picks = set(rng.choice(train_pairs, min(50, len(train_pairs)), replace=False))
    if train_singles:
        picks |= set(rng.choice(train_singles, min(20, len(train_singles)), replace=False))
    return {"val": val_monitor.reset_index(drop=True),
            "train": train[train.recipe_id.isin(picks)].reset_index(drop=True)}


def build_model(args, device):
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(BASE_WHISPER, attn_implementation="sdpa")
    model.config.apply_spec_augment = args.spec_augment > 0
    model.config.mask_time_prob = args.spec_augment
    model.config.mask_time_length = 10
    model.config.mask_feature_prob = 0.0
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=args.lora_dropout,
                      target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, lora)
    return model.to(device)


def lr_factor(step, warmup, total, floor=0.1):
    """Linear warm-up, then cosine decay to ``floor`` of the peak learning rate."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def epoch_batches(n, micro_batch, seed, epoch):
    order = np.random.default_rng(seed + 1000 * epoch).permutation(n)
    return [order[i:i + micro_batch].tolist() for i in range(0, n - micro_batch + 1, micro_batch)]


@torch.no_grad()
def mean_token_loss(model, frame, device, batch_size):
    """Teacher-forced cross-entropy per target token (no timestamps), for train/validation loss curves."""
    frame = frame[frame.text.notna()].reset_index(drop=True)
    data = TrackDataset(frame, timestamp_fraction=0.0)
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, len(data), batch_size):
        batch = collate([data[i] for i in range(start, min(start + batch_size, len(data)))])
        logits = model(input_features=batch["input_features"].to(device),
                       decoder_input_ids=batch["decoder_input_ids"].to(device)).logits
        labels = batch["labels"].to(device)
        mask = labels != -100
        total += float(torch.nn.functional.cross_entropy(logits[mask].float(), labels[mask], reduction="sum"))
        tokens += int(mask.sum())
    return total / max(tokens, 1)


def evaluate(model, sets, device, batch_size):
    result = {}
    for name, frame in sets.items():
        hypotheses = transcribe_hf(model, list(frame.path), device, batch_size)
        metrics, _ = score(frame, hypotheses)
        for kind, block in metrics.items():
            result[f"{name}_{kind}"] = block
    result["val_loss"] = mean_token_loss(model, sets["val"][sets["val"].kind == "pair"], device, batch_size)
    result["train_eval_loss"] = mean_token_loss(model, sets["train"][sets["train"].kind == "pair"], device, batch_size)
    return result


def save_state(model, optimizer, scheduler, run_dir, meta):
    from peft import get_peft_model_state_dict

    last = run_dir / "last"
    last.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "meta": meta,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}},
               last / "trainer_state.pt")


def load_state(model, optimizer, scheduler, run_dir):
    from peft import set_peft_model_state_dict

    state = torch.load(run_dir / "last" / "trainer_state.pt", map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    return state["meta"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--spec-augment", type=float, default=0.05, help="time-mask probability (0 disables)")
    parser.add_argument("--micro-batch", type=int, default=2, help="float32 on MPS needs ~3.3 GB per example")
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=0, help="also save resumable state every N steps (power loss)")
    parser.add_argument("--stop-at", type=int, default=None, help="pause after this optimizer step (schedule unchanged)")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                        help="bf16 autocast only on CUDA; MPS half-precision SDPA is numerically wrong")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="smaller fixed validation subset for LR pilots")
    parser.add_argument("--max-train-pairs", type=int, default=None, help="use only plan pairs below this index")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    if args.precision == "bf16" and device != "cuda":
        raise SystemExit("bf16 is only allowed on CUDA (see PIPELINE.md: MPS half-precision SDPA is wrong)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    train = training_tracks(args.max_train_pairs)
    sets = monitor_sets(train, args.seed, pilot=args.pilot or args.smoke)
    if args.smoke:
        train = train.sample(16, random_state=0).reset_index(drop=True)
        def tiny(frame):
            keep = sorted(frame[frame.kind == "pair"].recipe_id.unique())[:2] + sorted(frame[frame.kind == "single"].recipe_id.unique())[:1]
            return frame[frame.recipe_id.isin(keep)].reset_index(drop=True)
        sets = {k: tiny(v) for k, v in sets.items()}
        args.eval_every, args.stop_at, args.workers = 2, 2, 0
    effective = args.micro_batch * args.accumulate
    steps_per_epoch = len(train) // effective
    total_steps = max(1, int(round(args.epochs * steps_per_epoch)))
    config = {**{k: v for k, v in vars(args).items() if k not in ("resume", "stop_at")}, "device": device,
              "train_tracks": len(train), "train_speakers": int(train.speaker.nunique()),
              "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
              "monitor_tracks": {k: len(v) for k, v in sets.items()}}
    config_path = run_dir / "config.json"
    if args.resume:
        saved = json.loads(config_path.read_text())
        free = ("workers", "eval_every", "patience", "save_every", "eval_batch")  # do not change what is trained
        if {k: saved[k] for k in config if k not in free} != {k: config[k] for k in config if k not in free}:
            raise SystemExit("Resume configuration differs from the saved run (data or hyperparameters changed)")
    else:
        if (run_dir / "history.jsonl").exists():
            raise SystemExit(f"{run_dir} already has a history; use --resume or a new --run name")
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    model = build_model(args, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, args.warmup, total_steps))
    meta = {"step": 0, "best": None, "bad_evals": 0, "clean_reference": None}
    if args.resume:
        meta = load_state(model, optimizer, scheduler, run_dir)
    history = (run_dir / "history.jsonl").open("a")
    print(f"{args.run}: {len(train):,} tracks, {trainable:,} trainable of {total:,} parameters, "
          f"{total_steps} steps ({steps_per_epoch}/epoch) on {device}", flush=True)

    def log(record):
        history.write(json.dumps(record) + "\n")
        history.flush()

    def run_eval(step):
        started = time.monotonic()
        if device == "mps":
            torch.mps.empty_cache()
        result = evaluate(model, sets, device, args.eval_batch)
        if device == "mps":
            torch.mps.empty_cache()
        result.update(type="eval", step=step, epoch=round(step / max(steps_per_epoch, 1), 3),
                      seconds=round(time.monotonic() - started, 1))
        log(result)
        wer = lambda key: 100 * result.get(key, {}).get("wer", float("nan"))  # noqa: E731
        print(f"  eval step {step}: val mixtures {wer('val_pairs'):.2f}% · val clean {wer('val_singles'):.2f}% · "
              f"train mixtures {wer('train_pairs'):.2f}% · val loss {result['val_loss']:.3f} ({result['seconds']:.0f}s)", flush=True)
        return result

    if meta["step"] == 0 and meta["clean_reference"] is None:
        initial = RUNS / f"initial-eval-{'pilot' if args.pilot else 'smoke' if args.smoke else 'full'}.json"
        if initial.is_file():  # LoRA starts as an exact identity: step 0 is the pretrained model
            result = json.loads(initial.read_text())
            log(result)
        else:
            result = run_eval(0)
            initial.write_text(json.dumps(result) + "\n")
        meta["clean_reference"] = result["val_singles"]["wer"]
        # The pretrained model (step 0) is the bar to beat: best/ only exists if a checkpoint is better.
        meta["best"] = {"step": 0, "val_pairs_wer": result["val_pairs"]["wer"], "val_singles_wer": result["val_singles"]["wer"]}
        (run_dir / "best.json").write_text(json.dumps(meta["best"], indent=2) + "\n")

    def autocast():
        if args.precision == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()
    step, window, started = meta["step"], [], time.monotonic()
    stop = False
    while not stop and step < total_steps:
        epoch = step // max(steps_per_epoch, 1)
        batches = epoch_batches(len(train), args.micro_batch, args.seed, epoch)
        skip = (step - epoch * steps_per_epoch) * args.accumulate
        loader = torch.utils.data.DataLoader(TrackDataset(train), batch_sampler=batches[skip:], collate_fn=collate,
                                             num_workers=args.workers, persistent_workers=False,
                                             prefetch_factor=4 if args.workers else None)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                loss = model(**batch).loss / args.accumulate
            loss.backward()
            window.append(float(loss.detach()) * args.accumulate)
            if (micro + 1) % args.accumulate:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if device == "mps" and step % 5 == 0:
                torch.mps.empty_cache()  # the MPS caching allocator otherwise grows into swap (see PIPELINE.md)
            if step % 10 == 0 or step == total_steps:
                elapsed = time.monotonic() - started
                log({"type": "train", "step": step, "epoch": round(step / steps_per_epoch, 3),
                     "loss": float(np.mean(window)), "lr": scheduler.get_last_lr()[0],
                     "seconds_per_step": elapsed / 10 if step % 10 == 0 else None})
                print(f"step {step}/{total_steps} loss {np.mean(window):.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"{elapsed / 10:.1f}s/step", flush=True)
                window, started = [], time.monotonic()
            if args.save_every and step % args.save_every == 0 and step % args.eval_every:
                meta["step"] = step
                save_state(model, optimizer, scheduler, run_dir, meta)
            if step % args.eval_every == 0 or step == total_steps:
                result = run_eval(step)
                model.save_pretrained(run_dir / f"step-{step:05d}")
                eligible = result["val_singles"]["wer"] <= meta["clean_reference"] + CLEAN_TOLERANCE
                current = result["val_pairs"]["wer"]
                if eligible and current < meta["best"]["val_pairs_wer"] - MIN_IMPROVEMENT:
                    meta["best"] = {"step": step, "val_pairs_wer": current, "val_singles_wer": result["val_singles"]["wer"]}
                    meta["bad_evals"] = 0
                    model.save_pretrained(run_dir / "best")
                    (run_dir / "best.json").write_text(json.dumps(meta["best"], indent=2) + "\n")
                else:
                    meta["bad_evals"] += 1
                meta["step"] = step
                save_state(model, optimizer, scheduler, run_dir, meta)
                model.train()
                started = time.monotonic()
                if meta["bad_evals"] >= args.patience:
                    print(f"early stop at step {step}: no validation improvement in {args.patience} evaluations", flush=True)
                    stop = True
            if args.stop_at and step >= args.stop_at:
                meta["step"] = step
                save_state(model, optimizer, scheduler, run_dir, meta)
                print(f"paused at step {step} (--stop-at)", flush=True)
                stop = True
            if stop or step >= total_steps:
                break
    history.close()
    print(json.dumps({"run": args.run, "best": meta["best"], "last_step": step}), flush=True)


if __name__ == "__main__":
    main()
