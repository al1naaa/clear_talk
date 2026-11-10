"""LoRA fine-tuning of multilingual Whisper small on three languages and mixed recordings (steps 10-11).

    python -m cleartalk_ml.train --run main --lr 1.5e-4                    # full run with early stopping
    python -m cleartalk_ml.train --run pilot-lr5e-4 --lr 5e-4 --stop-at 100 --eval-every 100 --pilot
    python -m cleartalk_ml.train --run main --resume                        # continue after an interruption
    python -m cleartalk_ml.train --run smoke --smoke                        # tiny end-to-end check

Every ``--eval-every`` optimizer steps the model transcribes (greedy) a fixed validation subset - mixed
recordings and single utterances of every set - and a fixed subset of *training* examples, so the
train/validation gap is visible. The adapter with the lowest validation WER on mixed recordings whose
single-language FLEURS WER stays within the tolerance of the pretrained model is kept in ``best/``;
training stops after ``--patience`` evaluations without improvement. Resumable state is written every
``--save-every`` steps, so a power cut or a closed lid costs at most that many steps.
Test data are never opened here.
"""

import argparse
import contextlib
import json
import math
import random
import time

import numpy as np
import torch

from .dataset import ExampleDataset, collate, example_audio, features, processor, prompt_language
from .paths import BASES, LANGUAGES, MODELS, ROOT, RUNS, base_name
from .splits import SEP, composition_name, eval_units, mixed_example, training_examples
from .text import counts, pooled

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LANGUAGE_TOLERANCE = 0.02   # selection guard: FLEURS WER per language vs the pretrained model (absolute, greedy subset)
MIN_IMPROVEMENT = 0.001     # early-stopping threshold on validation WER (absolute)
MONITOR = {"full": {"mixed": 80, "single": 40, "train_mixed": 40, "train_single": 12},
           "pilot": {"mixed": 60, "single": 30, "train_mixed": 30, "train_single": 10}}


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def validation_examples():
    """Validation units as examples (same format as training examples), with their references."""
    from .corpora import load_manifest
    from .mixed import load_plan
    from .splits import cleaned

    units = eval_units("val")
    table = cleaned(load_manifest()).set_index("utt_id")
    recipes = {recipe["mix_id"]: recipe for recipe in load_plan("val")}
    examples = []
    for unit in units.itertuples():
        if unit.kind == "mixed":
            example = mixed_example(recipes[unit.unit_id], {})
        else:
            row = table.loc[unit.unit_id]
            example = {"id": unit.unit_id, "kind": "single", "corpus": row.corpus, "language": row.language,
                       "path": row.path, "duration": float(row.duration), "text": row.text}
        examples.append({**example, "set": unit.set, "reference": unit.reference.replace(SEP, " ")})
    return examples


def monitor_sets(train, seed, sizes):
    """Fixed evaluation subsets, chosen by seed only (never by results)."""
    rng = np.random.default_rng(seed)

    def pick(items, count):
        items = sorted(items, key=lambda e: e["id"])
        return [items[i] for i in sorted(rng.permutation(len(items))[:count])]

    val = validation_examples()
    val_monitor = []
    for name in sorted({e["set"] for e in val}):
        group = [e for e in val if e["set"] == name and e["duration"] <= 30]
        val_monitor += pick(group, sizes["mixed"] if name == "mixed" else sizes["single"])
    train = [{**e, "set": "mixed" if e["kind"] == "mixed" else f"{e['corpus']}-{e['language']}",
              "reference": " ".join(p["text"] for p in e["parts"]) if e["kind"] == "mixed" else e["text"]} for e in train]
    train_monitor = []
    for name in sorted({e["set"] for e in train}):
        group = [e for e in train if e["set"] == name]
        train_monitor += pick(group, sizes["train_mixed"] if name == "mixed" else sizes["train_single"])
    return {"val": val_monitor, "train": train_monitor}


def build_model(args, device):
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    from transformers import GenerationConfig

    base = BASES[base_name()]
    model = WhisperForConditionalGeneration.from_pretrained(MODELS / base["hf"], dtype=torch.float32, attn_implementation="sdpa")
    # Generation settings (language and task tokens, suppressed tokens) come from the processor folder.
    model.generation_config = GenerationConfig.from_pretrained(MODELS / base["processor"])
    model.config.apply_spec_augment = args.spec_augment > 0
    model.config.mask_time_prob = args.spec_augment
    model.config.mask_time_length = 10
    model.config.mask_feature_prob = 0.0
    targets = LORA_TARGETS
    if args.encoder_layers:  # round 5: the decoder adapter plus LoRA on the attention of the top encoder layers
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        count = model.config.encoder_layers
        targets = [name for name, _ in model.named_modules()
                   if name.startswith("model.decoder.layers") and name.split(".")[-1] in LORA_TARGETS]
        targets += [f"model.encoder.layers.{i}.self_attn.{proj}" for i in range(count - args.encoder_layers, count) for proj in ("q_proj", "v_proj")]
        peft_model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=args.lora_dropout,
                                                      target_modules=targets, bias="none"))
        if args.init_adapter:  # the decoder starts from the trained adapter; the new encoder adapters start at zero
            loaded = set_peft_model_state_dict(peft_model, load_file(str(ROOT / args.init_adapter / "adapter_model.safetensors")))
            if loaded.unexpected_keys:
                raise RuntimeError(f"adapter keys that do not fit: {loaded.unexpected_keys[:3]}")
        # The frozen lower layers keep no activations; the trained top layers are recomputed in the backward pass.
        peft_model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return peft_model.to(device)
    if base["lora"] == "decoder":  # adapters only in the decoder: the encoder is frozen and needs no gradients
        targets = [name for name, _ in model.named_modules()
                   if name.startswith("model.decoder.layers") and name.split(".")[-1] in LORA_TARGETS]
    if args.init_adapter:  # continue from a trained adapter (its rank and targets come with it)
        from peft import PeftModel

        return PeftModel.from_pretrained(model, ROOT / args.init_adapter, is_trainable=True).to(device)
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=args.lora_dropout,
                      target_modules=targets, bias="none")
    return get_peft_model(model, lora).to(device)


def lr_factor(step, warmup, total, floor=0.1):
    """Linear warm-up, then cosine decay to ``floor`` of the peak learning rate."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def epoch_batches(n, micro_batch, seed, epoch):
    order = np.random.default_rng(seed + 1000 * epoch).permutation(n)
    return [order[i:i + micro_batch].tolist() for i in range(0, n - micro_batch + 1, micro_batch)]


def release_graphics_memory(device):
    """Give PyTorch's cached, unused blocks back to the graphics driver.

    Training leaves several gigabytes cached. An evaluation allocates blocks of other sizes; on a 6 GB card the
    total then passes the card's memory, Windows spills into shared memory without an error, and the evaluation
    crawls (round 2: one took 2 h 8 min instead of 9 min). Changes no number: it is memory bookkeeping only.
    """
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


@torch.no_grad()
def transcribe_greedy(model, examples, device, batch_size):
    """Greedy transcripts without timestamps; each example is prompted with its language token."""
    tokenizer = processor().tokenizer
    texts = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        batch = np.stack([features(example_audio(example)) for example in chunk])
        generated = model.generate(input_features=torch.from_numpy(batch).to(device), task="transcribe",
                                   language=[prompt_language(example) for example in chunk],
                                   max_new_tokens=300, num_beams=1, do_sample=False, return_timestamps=False)
        texts.extend(text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True))
        del generated
        release_graphics_memory(device)
    return texts


@torch.no_grad()
def mean_token_loss(model, examples, device, batch_size):
    """Teacher-forced cross-entropy per target token (no timestamps), for train/validation loss curves."""
    data = ExampleDataset(examples, timestamp_fraction=0.0)
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
    """WER per set for the validation and training monitors, plus the loss on mixed recordings."""
    result = {}
    release_graphics_memory(device)
    for name, examples in sets.items():
        hypotheses = transcribe_greedy(model, examples, device, batch_size)
        rows = {}
        for example, hypothesis in zip(examples, hypotheses):
            rows.setdefault(example["set"], []).append(counts(example["reference"], hypothesis))
        for set_name, items in rows.items():
            result[f"{name}_{set_name}"] = {**pooled(items), "units": len(items)}
        result[f"{name}_loss"] = mean_token_loss(model, [e for e in examples if e["set"] == "mixed"], device, batch_size)
        release_graphics_memory(device)
    return result


def save_state(model, optimizer, scheduler, run_dir, meta):
    from peft import get_peft_model_state_dict

    last = run_dir / "last"
    last.mkdir(parents=True, exist_ok=True)
    temporary = last / "trainer_state.tmp"
    torch.save({"adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "meta": meta,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}},
               temporary)
    temporary.replace(last / "trainer_state.pt")  # a crash while writing never leaves a broken state file


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
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--spec-augment", type=float, default=0.05, help="time-mask probability (0 disables)")
    parser.add_argument("--base", choices=sorted(BASES), default="small",
                        help="base model: small (LoRA everywhere) or kazakh-turbo (LoRA in the decoder only)")
    parser.add_argument("--micro-batch", type=int, default=None, help="default: the base model's setting")
    parser.add_argument("--accumulate", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--mixed", type=int, default=None, help="mixed recordings per epoch (default: all planned)")
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=25, help="also save resumable state every N steps")
    parser.add_argument("--stop-at", type=int, default=None, help="pause after this optimizer step (schedule unchanged)")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32",
                        help="reduced precision only on CUDA and only after scripts/check_precision.py agrees with fp32")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="smaller fixed validation subset for LR pilots")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--noise-prob", type=float, default=0.0, help="round 4: share of training examples with background noise")
    parser.add_argument("--encoder-layers", type=int, default=0, help="round 5: also LoRA on the attention of this many top encoder layers")
    parser.add_argument("--init-adapter", default=None, help="start from these LoRA weights (a folder under the project) instead of new ones")
    return parser.parse_args()


def main():
    from .awake import keep_awake

    import os

    args = parse_args()
    os.environ["CLEARTALK_BASE"] = args.base  # read by the dataset code here and in the data-loading workers
    args.micro_batch = args.micro_batch or BASES[args.base]["micro_batch"]
    args.accumulate = args.accumulate or BASES[args.base]["accumulate"]
    keep_awake()
    device = pick_device(args.device)
    if args.precision != "fp32" and device != "cuda":
        raise SystemExit("Reduced precision is only allowed on CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    train = sorted(training_examples(args.mixed), key=lambda e: e["id"])
    sets = monitor_sets(train, args.seed, MONITOR["pilot" if args.pilot or args.smoke else "full"])
    talkers = []
    if args.noise_prob > 0:
        from .noise import talker_pool

        talkers = talker_pool()
        # A noisy copy of the mixed validation monitor (babble at 10 dB), so the curves show robustness as well.
        sets["valnoise"] = [{**e, "noise": {"kind": "babble", "snr": 10.0, "seed": 7000 + k, "talkers": talkers[:300]}}
                            for k, e in enumerate(x for x in sets["val"] if x["set"] == "mixed")]
    if args.smoke:
        picks = np.random.default_rng(0).permutation(len(train))[:16]
        train = [train[i] for i in sorted(picks)]
        sets = {name: [e for k, e in enumerate(items) if k % max(1, len(items) // 6) == 0][:8] for name, items in sets.items()}
        args.eval_every, args.stop_at, args.workers, args.save_every = 2, 2, 0, 1
    effective = args.micro_batch * args.accumulate
    steps_per_epoch = len(train) // effective
    total_steps = max(1, int(round(args.epochs * steps_per_epoch)))
    config = {**{k: v for k, v in vars(args).items() if k not in ("resume", "stop_at")}, "device": device, "data": composition_name(),
              "train_examples": len(train), "train_mixed": sum(e["kind"] == "mixed" for e in train),
              "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
              "monitor_units": {name: len(items) for name, items in sets.items()}}
    config_path = run_dir / "config.json"
    if args.resume:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        saved.setdefault("base", "small")  # runs made before other base models existed
        saved.setdefault("data", "base")   # runs made before other training compositions existed
        saved.setdefault("noise_prob", 0.0)
        saved.setdefault("init_adapter", None)
        saved.setdefault("encoder_layers", 0)
        free = ("workers", "eval_every", "patience", "save_every", "eval_batch")  # do not change what is trained
        if {k: saved[k] for k in config if k not in free} != {k: config[k] for k in config if k not in free}:
            raise SystemExit("Resume configuration differs from the saved run (data or hyperparameters changed)")
    else:
        if (run_dir / "history.jsonl").exists():
            raise SystemExit(f"{run_dir} already has a history; use --resume or a new --run name")
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    model = build_model(args, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, args.warmup, total_steps))
    meta = {"step": 0, "best": None, "bad_evals": 0, "reference": None}
    if args.resume:
        meta = load_state(model, optimizer, scheduler, run_dir)
    history = (run_dir / "history.jsonl").open("a", encoding="utf-8")
    print(f"{args.run}: {len(train):,} examples, {trainable:,} trainable of {total:,} parameters, "
          f"{total_steps} steps ({steps_per_epoch}/epoch) on {device} ({args.precision})", flush=True)

    def log(record):
        history.write(json.dumps(record) + "\n")
        history.flush()

    def fleurs(result):
        return {language: result[f"val_fleurs-{language}"]["wer"] for language in LANGUAGES if f"val_fleurs-{language}" in result}

    def run_eval(step):
        started = time.monotonic()
        result = evaluate(model, sets, device, args.eval_batch)
        result.update(type="eval", step=step, epoch=round(step / max(steps_per_epoch, 1), 3),
                      seconds=round(time.monotonic() - started, 1))
        log(result)
        wer = lambda key: 100 * result.get(key, {}).get("wer", float("nan"))  # noqa: E731
        print(f"  eval step {step}: val mixed {wer('val_mixed'):.2f}% · kk {wer('val_fleurs-kk'):.2f}% · ru {wer('val_fleurs-ru'):.2f}% · "
              f"en {wer('val_fleurs-en'):.2f}% · ksc2 {wer('val_ksc2-kk'):.2f}% · golos {wer('val_golos-ru'):.2f}% · "
              f"train mixed {wer('train_mixed'):.2f}% · val loss {result['val_loss']:.3f} ({result['seconds']:.0f}s)", flush=True)
        return result

    if meta["step"] == 0 and meta["reference"] is None:
        mode = "smoke" if args.smoke else "pilot" if args.pilot else "full"
        # The training-subset monitor depends on the composition, so each composition caches its own step 0.
        tag = "" if args.base == "small" else f"{args.base}-"
        tag += "" if composition_name() == "base" else f"{composition_name()}-"
        initial = RUNS / f"initial-eval-{tag}{mode}.json"
        if args.init_adapter:  # a warm start: step 0 is the trained adapter, measured afresh
            result = run_eval(0)
        elif initial.is_file():  # LoRA starts as an exact identity: step 0 is the pretrained model
            result = json.loads(initial.read_text(encoding="utf-8"))
            log(result)
        else:
            result = run_eval(0)
            initial.write_text(json.dumps(result) + "\n", encoding="utf-8")
        meta["reference"] = fleurs(result)
        # The pretrained model (step 0) is the bar to beat: best/ only exists if a checkpoint is better.
        meta["best"] = {"step": 0, "val_mixed_wer": result["val_mixed"]["wer"], "val_fleurs_wer": fleurs(result)}
        (run_dir / "best.json").write_text(json.dumps(meta["best"], indent=2) + "\n", encoding="utf-8")

    def autocast():
        if args.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16)

    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    step, window, started = meta["step"], [], time.monotonic()
    stop = False
    while not stop and step < total_steps:
        epoch = step // max(steps_per_epoch, 1)
        batches = epoch_batches(len(train), args.micro_batch, args.seed, epoch)
        skip = (step - epoch * steps_per_epoch) * args.accumulate
        loader = torch.utils.data.DataLoader(ExampleDataset(train, noise_probability=args.noise_prob, talkers=talkers),
                                             batch_sampler=batches[skip:], collate_fn=collate,
                                             num_workers=args.workers, persistent_workers=False,
                                             prefetch_factor=4 if args.workers else None)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast():
                loss = model(**batch).loss / args.accumulate
            scaler.scale(loss).backward()
            window.append(float(loss.detach()) * args.accumulate)
            if (micro + 1) % args.accumulate:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
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
                eligible = all(wer <= meta["reference"][language] + LANGUAGE_TOLERANCE for language, wer in fleurs(result).items())
                current = result["val_mixed"]["wer"]
                if eligible and current < meta["best"]["val_mixed_wer"] - MIN_IMPROVEMENT:
                    meta["best"] = {"step": step, "val_mixed_wer": current, "val_fleurs_wer": fleurs(result)}
                    meta["bad_evals"] = 0
                    model.save_pretrained(run_dir / "best")
                    (run_dir / "best.json").write_text(json.dumps(meta["best"], indent=2) + "\n", encoding="utf-8")
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
    summary = {"run": args.run, "best": meta["best"], "last_step": step, "total_steps": total_steps,
               "finished": bool(step >= total_steps or meta["bad_evals"] >= args.patience)}
    if summary["finished"]:  # a pause (--stop-at) is not the end of the run
        (run_dir / "finished.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
