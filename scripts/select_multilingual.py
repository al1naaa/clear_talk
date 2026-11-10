"""Choose the checkpoint on validation, export it for the app, apply the promotion rules - resumable.

    python scripts/select_multilingual.py

1. The last checkpoint and the one the greedy training monitor liked best are exported and scored on the
   mixed validation recordings through the app's decoding path. The lower WER wins.
2. The winner is scored on the remaining validation sets (FLEURS, Golos, KSC2).
3. Rule 1 of the contract (against untouched Whisper small) decides whether it is promoted.
4. It is exported to models/multilingual/whisper-small-cleartalk-multilingual with a model card.
5. Rule 2 of the contract writes the app's routes (which recognizer serves which language) to
   models/multilingual/profile.json, with SHA-256 signatures of every installed recognizer.

Test data are not touched. Every transcription is cached, so an interrupted run continues.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from backend.config import DEFAULT_MULTILINGUAL_PROFILE
    from cleartalk_ml import validation
    from cleartalk_ml.convert import sha256
    from cleartalk_ml.evaluate import hyp_path, pick_device, run_units, units_of
    from cleartalk_ml.export import EXPORT, export
    from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES, MODELS, RESULTS, RUNS

    choice = json.loads((RUNS / "lr-choice.json").read_text(encoding="utf-8"))
    run = choice["chosen"]
    run_dir = RUNS / run
    if not (run_dir / "finished.json").is_file():
        raise SystemExit("Training has not finished; run scripts/run_training.py first")
    finished = json.loads((run_dir / "finished.json").read_text(encoding="utf-8"))
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    steps = sorted(int(p.name.split("-")[1]) for p in run_dir.glob("step-*") if (p / "adapter_model.safetensors").is_file())
    shortlist = sorted({steps[-1], int(finished["best"]["step"])} & set(steps)) or [steps[-1]]
    device, precision = pick_device()

    def system_of(step):
        return f"runs/{run}/ct2-step-{step:05d}"

    def score(step, sets):
        output = MODELS / system_of(step)
        if not (output / "model_card.json").is_file():
            export(run_dir / f"step-{step:05d}", output, {"step": step})
        for set_name in sets:
            run_units(system_of(step), units_of(set_name), hyp_path(set_name, system_of(step), device, precision), device, precision)

    # Whisper small must be scored on the same sets on the same backend (cached when already done).
    for set_name in ("mixed-val", "fleurs-val", "golos-val", "ksc2-val"):
        run_units("small", units_of(set_name), hyp_path(set_name, "small", device, precision), device, precision)

    candidates = {}
    for step in shortlist:
        score(step, ["mixed-val"])
        candidates[step] = validation.mixed_scores(system_of(step))["wer"]
        print(f"step {step}: mixed validation WER {100 * candidates[step]:.2f}%", flush=True)
    order = sorted(candidates, key=lambda step: (round(candidates[step], 5), -step))
    decision = None
    for step in order:  # the best checkpoint that keeps every language (rule 1b); otherwise the best on mixed
        score(step, ["fleurs-val", "golos-val", "ksc2-val"])
        outcome = validation.promotion(system_of(step))
        if decision is None or (outcome["rule_b_no_language_worse"] and not decision["rule_b_no_language_worse"]):
            decision = {**outcome, "step": step}
        if outcome["rule_b_no_language_worse"]:
            break
    step, system = decision["step"], system_of(decision["step"])
    small_mixed, ours_mixed = validation.mixed_scores("small"), validation.mixed_scores(system)
    small_single, ours_single = validation.single_scores("small"), validation.single_scores(system)

    percent = lambda value: f"{100 * value:.1f}%"  # noqa: E731
    trainable = 6488064
    evidence = [
        "ClearTalk multilingual: Whisper small adapted with LoRA to Kazakh, Russian, English and to recordings that change language.",
        f"{trainable:,} trained LoRA parameters ({100 * trainable / 248222976:.1f}% of 248,222,976); merged, so it runs as fast as Whisper small.",
        f"Trained on {config['train_examples']:,} examples ({config['train_mixed']:,} of them mixed-language recordings assembled from "
        "FLEURS, Golos and KSC2 utterances), one pass, on a GTX 1660 Ti.",
        f"Validation, mixed-language recordings ({ours_mixed['units']}): word error rate {percent(small_mixed['wer'])} Whisper small → "
        f"{percent(ours_mixed['wer'])} ClearTalk. Lower is better.",
    ]
    for language in LANGUAGES:
        key = f"fleurs-{language}"
        evidence.append(f"Validation, {LANGUAGE_NAMES[language]} read sentences ({ours_single[key]['units']}): "
                        f"{percent(small_single[key]['forced']['wer'])} Whisper small → {percent(ours_single[key]['forced']['wer'])} ClearTalk.")
    for key, label in (("ksc2-kk", "Kazakh broadcast and read speech"), ("golos-ru", "short Russian phrases")):
        evidence.append(f"Validation, {label} ({ours_single[key]['units']}): {percent(small_single[key]['forced']['wer'])} Whisper small → "
                        f"{percent(ours_single[key]['forced']['wer'])} ClearTalk.")
    evidence.append("Mixed recordings are assembled from single-language utterances: every language change is also a change of "
                    "speaker. Real conversations, accents and microphones can behave differently.")
    card = {
        "model_name": "ClearTalk multilingual Whisper small (LoRA fine-tuned)", "run": run, "step": step,
        "trainable_parameters": trainable, "total_parameters": 248222976,
        "training": {"examples": config["train_examples"], "mixed_recordings": config["train_mixed"], "learning_rate": config["lr"],
                     "optimizer_steps": finished["last_step"]},
        "validation": {"mixed": {"baseline_wer": small_mixed["wer"], "cleartalk_wer": ours_mixed["wer"], "recordings": ours_mixed["units"]},
                       **{key: {"baseline_wer": small_single[key]["forced"]["wer"], "cleartalk_wer": ours_single[key]["forced"]["wer"],
                                "utterances": ours_single[key]["units"]} for key in ours_single}},
        "promotion": {k: v for k, v in decision.items() if k != "step"},
        "evidence": evidence,
    }
    # The app's model is a copy of the very folder that was validated (not a second conversion), plus its card.
    if EXPORT.is_dir():
        shutil.rmtree(EXPORT)
    shutil.copytree(MODELS / system, EXPORT)
    card.update(adapter=f"models/multilingual/runs/{run}/step-{step:05d}", base_model="openai/whisper-small",
                files_sha256={p.name: sha256(p) for p in sorted(EXPORT.iterdir()) if p.is_file() and p.name != "model_card.json"})
    (EXPORT / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    same = sha256(EXPORT / "model.bin") == sha256(MODELS / system / "model.bin")
    print(f"exported step {step}; identical to the validated file: {same}", flush=True)
    if not same:
        raise SystemExit("The exported model differs from the validated checkpoint")

    # Rule 2: routes. Every candidate recognizer must be scored on every set of its language first
    # (cached when already done), otherwise a recognizer would win a language by default.
    for reference, sets in (("turbo", ("fleurs-val", "golos-val", "ksc2-val")), ("kazakh", ("fleurs-val", "ksc2-val"))):
        for set_name in sets:
            run_units(reference, units_of(set_name), hyp_path(set_name, reference, device, precision), device, precision)
    systems = {"turbo": "turbo", "kazakh": "kazakh", "small": "small", "cleartalk": system}
    profile = json.loads(json.dumps(DEFAULT_MULTILINGUAL_PROFILE))
    routing = validation.routes(systems, {key: entry["languages"] for key, entry in profile["recognizers"].items()})
    for language in LANGUAGES:
        if language in routing["routes"]:
            profile["routes"][language] = routing["routes"][language]
    # Rule 1: which small recognizer stands in when the large models are not installed.
    profile["small_recognizer"] = "cleartalk" if decision["promoted"] else "small"
    for key, entry in profile["recognizers"].items():
        directory = MODELS / entry["directory"]
        if (directory / "model.bin").is_file():
            entry["files_sha256"] = {name: sha256(directory / name) for name in ("model.bin", "config.json", "tokenizer.json")}
    (MODELS / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {"run": run, "checkpoints_scored_on_mixed_validation": {str(k): v for k, v in candidates.items()}, "chosen_step": step,
               "promotion": {k: v for k, v in decision.items() if k != "step"}, "routes": routing, "exported_identical": same}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "selection.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"chosen_step": step, "promoted": decision["promoted"], "routes": routing["routes"]}, indent=1))
    print("SELECTION_DONE", flush=True)
    # Reference points for the validation report (they decide nothing): the large model in one pass and the
    # app's Conversation pipeline with the routes just written, on the mixed validation recordings.
    for reference in ("turbo", "pipeline:auto"):
        run_units(reference, units_of("mixed-val"), hyp_path("mixed-val", reference, device, precision), device, precision)
    print("REFERENCES_DONE", flush=True)


if __name__ == "__main__":
    main()
