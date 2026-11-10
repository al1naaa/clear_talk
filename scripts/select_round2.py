"""Round 2: choose the checkpoint of the large model on validation, export it, apply the pre-registered rules.

    python scripts/select_round2.py

Everything is decided on validation data; test B stays untouched. Every transcription is cached, so an
interrupted run continues with the same command.

1. The last checkpoint and the one the greedy training monitor liked best are exported and scored on the
   mixed validation recordings through the app's decoding path. The lower WER wins (a tie: the later step).
2. The winner is scored on the remaining validation sets. The untouched starting model (the Kazakh turbo
   fine-tune) is scored on every set too: it is the baseline of this round.
3. The winner is copied to models/multilingual/whisper-large-cleartalk-multilingual with a model card.
4. Rule 1 (routes): each language goes to the installed recognizer with the lowest pooled validation WER
   over that language's sets.
5. Rule 2 (recognizer for mixed speech): the recognizer that covers all three languages and has the lowest
   one-pass WER on mixed validation recordings; an exact tie goes to the smaller model.
6. Report, not a rule: the new model against its untouched starting model per language; a Kazakh loss of
   more than 0.5 points is flagged.
7. models/multilingual/profile.json is rewritten; the app's pipeline is then measured on the mixed
   validation recordings with the new profile (reference points, they decide nothing).
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "kazakh-turbo"
VALIDATION_SETS = ("mixed-val", "fleurs-val", "golos-val", "ksc2-val")
TRAINABLE, TOTAL = 2_129_920, 811_008_000      # printed by cleartalk_ml.train for this base
SIZE = {"small": 1, "cleartalk": 1, "turbo": 2, "kazakh": 2, "cleartalk-large": 2}   # for "a tie goes to the smaller model"
KAZAKH_TOLERANCE = 0.005
SINGLE = (("fleurs-kk", "Kazakh read sentences"), ("fleurs-ru", "Russian read sentences"), ("fleurs-en", "English read sentences"),
          ("ksc2-kk", "Kazakh broadcast and read speech"), ("golos-ru", "short Russian phrases"))


def main():
    from backend.config import DEFAULT_MULTILINGUAL_PROFILE
    from cleartalk_ml import validation
    from cleartalk_ml.awake import keep_awake
    from cleartalk_ml.convert import sha256
    from cleartalk_ml.evaluate import bootstrap_difference, hyp_path, pick_device, run_units, units_of
    from cleartalk_ml.export import export
    from cleartalk_ml.paths import BASES, LANGUAGES, MODELS, RESULTS, RUNS

    keep_awake()
    target = MODELS / BASES[BASE]["export"]
    choice = json.loads((RUNS / "large-lr-choice.json").read_text(encoding="utf-8"))
    run = choice["chosen"]
    run_dir = RUNS / run
    if not (run_dir / "finished.json").is_file():
        raise SystemExit("Training has not finished; run scripts/run_training.py --base kazakh-turbo first")
    finished = json.loads((run_dir / "finished.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    steps = sorted(int(p.name.split("-")[1]) for p in run_dir.glob("step-*") if (p / "adapter_model.safetensors").is_file())
    shortlist = sorted({steps[-1], int(finished["best"]["step"])} & set(steps)) or [steps[-1]]
    device, precision = pick_device()
    round1 = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    small_ours = f"runs/{round1['run']}/ct2-step-{round1['chosen_step']:05d}"   # the round-1 model as it was validated

    def system_of(step):
        return f"runs/{run}/ct2-step-{step:05d}"

    def score(system, sets):
        for set_name in sets:
            run_units(system, units_of(set_name), hyp_path(set_name, system, device, precision), device, precision)

    def exported(step):
        output = MODELS / system_of(step)
        if not (output / "model_card.json").is_file():
            export(run_dir / f"step-{step:05d}", output, {"step": step}, base=BASE)
        return system_of(step)

    # 1. checkpoint by the app's decoding on mixed validation recordings
    candidates = {}
    for step in shortlist:
        score(exported(step), ["mixed-val"])
        candidates[step] = validation.mixed_scores(system_of(step))["wer"]
        print(f"step {step}: mixed validation WER {100 * candidates[step]:.2f}%", flush=True)
    step = min(candidates, key=lambda item: (round(candidates[item], 5), -item))
    system = system_of(step)

    # 2. the winner and the untouched starting model on every validation set; the other recognizers where missing
    score(system, VALIDATION_SETS)
    score("kazakh", VALIDATION_SETS)
    for reference in ("turbo", "small", small_ours):
        score(reference, VALIDATION_SETS)
    print("VALIDATION_SCORED", flush=True)

    mixed = {name: validation.mixed_scores(key) for name, key in
             {"cleartalk-large": system, "kazakh": "kazakh", "turbo": "turbo", "small": "small", "cleartalk": small_ours}.items()}
    single = {name: validation.single_scores(key) for name, key in
              {"cleartalk-large": system, "kazakh": "kazakh", "turbo": "turbo", "small": "small", "cleartalk": small_ours}.items()}

    def interval(a, b):
        return {k: v for k, v in bootstrap_difference(mixed[a]["details"], mixed[b]["details"]).items()}

    # 6. the new model against its untouched starting model, per language (a report, not a rule)
    against_start = {}
    for key, _ in SINGLE:
        before, after = single["kazakh"][key]["forced"]["wer"], single["cleartalk-large"][key]["forced"]["wer"]
        against_start[key] = {"starting_model_wer": before, "cleartalk_large_wer": after, "change": after - before,
                              "units": single["cleartalk-large"][key]["units"],
                              "language_accuracy": single["cleartalk-large"][key]["language_accuracy"]}
    kazakh_worse = {key: against_start[key]["change"] for key in ("fleurs-kk", "ksc2-kk") if against_start[key]["change"] > KAZAKH_TOLERANCE}

    # 3. the app's model: a copy of the very folder that was validated, plus its card
    percent = lambda value: f"{100 * value:.1f}%"  # noqa: E731
    evidence = [
        "ClearTalk large multilingual: the Kazakh Whisper large-v3-turbo adapted with LoRA (decoder only) to Kazakh, Russian, "
        "English and to recordings that change language.",
        f"{TRAINABLE:,} trained LoRA parameters ({100 * TRAINABLE / TOTAL:.1f}% of {TOTAL:,}); merged, so it runs as fast as "
        "Whisper large-v3-turbo.",
        f"Trained on {config['train_examples']:,} examples ({config['train_mixed']:,} of them mixed-language recordings assembled from "
        "FLEURS, Golos and KSC2 utterances), one pass, on a GTX 1660 Ti.",
        f"Validation, mixed-language recordings ({mixed['cleartalk-large']['units']}): word error rate {percent(mixed['kazakh']['wer'])} "
        f"Kazakh Whisper large-v3-turbo → {percent(mixed['cleartalk-large']['wer'])} ClearTalk large. Lower is better.",
        f"Validation, mixed-language recordings ({mixed['cleartalk-large']['units']}): {percent(mixed['cleartalk']['wer'])} ClearTalk small → "
        f"{percent(mixed['cleartalk-large']['wer'])} ClearTalk large.",
    ]
    for key, label in SINGLE:
        evidence.append(f"Validation, {label} ({against_start[key]['units']}): {percent(against_start[key]['starting_model_wer'])} "
                        f"Kazakh Whisper large-v3-turbo → {percent(against_start[key]['cleartalk_large_wer'])} ClearTalk large.")
    evidence.append("Mixed recordings are assembled from single-language utterances: every language change is also a change of "
                    "speaker. Real conversations, accents and microphones can behave differently.")
    card = {
        "model_name": "ClearTalk large multilingual (Kazakh Whisper large-v3-turbo, decoder LoRA)", "run": run, "step": step,
        "trainable_parameters": TRAINABLE, "total_parameters": TOTAL,
        "training": {"examples": config["train_examples"], "mixed_recordings": config["train_mixed"], "learning_rate": config["lr"],
                     "optimizer_steps": finished["last_step"], "lora": "decoder only, rank 16"},
        "validation": {"mixed": {"baseline_wer": mixed["kazakh"]["wer"], "cleartalk_wer": mixed["cleartalk-large"]["wer"],
                                 "recordings": mixed["cleartalk-large"]["units"]},
                       **{key: {"baseline_wer": item["starting_model_wer"], "cleartalk_wer": item["cleartalk_large_wer"],
                                "utterances": item["units"]} for key, item in against_start.items()}},
        "baseline": "the untouched starting model (shyngys879/kazakh-whisper-large-v3-turbo)",
        "kazakh_worse_than_starting_model": kazakh_worse,
        "evidence": evidence,
    }
    if target.is_dir():
        shutil.rmtree(target)
    shutil.copytree(MODELS / system, target)
    card.update(adapter=f"models/multilingual/runs/{run}/step-{step:05d}", base_model=BASES[BASE]["name"],
                files_sha256={p.name: sha256(p) for p in sorted(target.iterdir()) if p.is_file() and p.name != "model_card.json"})
    (target / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    same = sha256(target / "model.bin") == sha256(MODELS / system / "model.bin")
    print(f"exported step {step}; identical to the validated file: {same}", flush=True)
    if not same:
        raise SystemExit("The exported model differs from the validated checkpoint")

    # 4. rule 1: routes
    systems = {"turbo": "turbo", "kazakh": "kazakh", "small": "small", "cleartalk": small_ours, "cleartalk-large": system}
    profile = json.loads(json.dumps(DEFAULT_MULTILINGUAL_PROFILE))
    covers = {key: entry["languages"] for key, entry in profile["recognizers"].items()}
    routing = validation.routes(systems, covers)
    for language in LANGUAGES:
        if language in routing["routes"]:
            profile["routes"][language] = routing["routes"][language]

    # 5. rule 2: the recognizer for recordings whose languages change inside turns
    one_pass = {key: mixed[key]["wer"] for key in systems if set(covers[key]) == set(LANGUAGES)}
    profile["mixed_recognizer"] = min(one_pass, key=lambda key: (round(one_pass[key], 5), SIZE[key], key))

    current = MODELS / "profile.json"
    if current.is_file():  # rule 1 of round 1 stays as it was decided
        profile["small_recognizer"] = json.loads(current.read_text(encoding="utf-8")).get("small_recognizer", profile["small_recognizer"])
    for key, entry in profile["recognizers"].items():
        directory = MODELS / entry["directory"]
        if (directory / "model.bin").is_file():
            entry["files_sha256"] = {name: sha256(directory / name) for name in ("model.bin", "config.json", "tokenizer.json")}
    current.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "run": run, "checkpoints_scored_on_mixed_validation": {str(k): v for k, v in candidates.items()}, "chosen_step": step,
        "exported_identical": same,
        "rule_1_routes": routing,
        "rule_2_mixed_recognizer": {"chosen": profile["mixed_recognizer"], "one_pass_mixed_validation_wer": one_pass,
                                    "large_minus_small_ours": interval("cleartalk", "cleartalk-large"),
                                    "large_minus_starting_model": interval("kazakh", "cleartalk-large")},
        "report_against_starting_model": against_start,
        "kazakh_worse_than_starting_model_by_more_than_half_a_point": kazakh_worse,
        "mixed_by_language": {key: mixed[key]["by_language"] for key in mixed},
    }
    (RESULTS / "selection-round2.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"chosen_step": step, "routes": routing["routes"], "mixed_recognizer": profile["mixed_recognizer"],
                      "one_pass_mixed": {k: round(v, 4) for k, v in one_pass.items()}, "kazakh_worse": kazakh_worse}, indent=1))
    print("SELECTION_DONE", flush=True)

    # 7. reference points with the profile just written: the app's pipeline on the mixed validation recordings
    for reference, name in (("pipeline:auto", "pipeline-auto-r2"), ("pipeline:mixed", "pipeline-mixed-r2")):
        run_units(reference, units_of("mixed-val"), RESULTS / "hyp" / f"mixed-val__{name}__{device}-{precision}.jsonl", device, precision)
    print("REFERENCES_DONE", flush=True)


if __name__ == "__main__":
    main()
