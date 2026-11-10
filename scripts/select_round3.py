"""Round 3: choose the checkpoint of the small model trained with Kazakh broadcast speech, export it, apply the rules.

    python scripts/select_round3.py

Everything is decided on validation data; test C stays untouched. Transcriptions are cached, so an interrupted run
continues with the same command. The rules are those written in ml_pipeline/round3/PIPELINE.md before training:

* Checkpoint: of the last checkpoint and the greedy monitor's best, the lowest pooled Kazakh validation WER among
  those that meet guards (b) and (c) of rule 1; if none does, the lowest pooled Kazakh WER.
* Rule 1 (promotion to the app's small recognizer, against the round-1 model): (a) pooled Kazakh WER (FLEURS kk +
  KSC2) at least 10% lower, relative, with the 95% bootstrap interval of the difference below zero; (b) one-pass WER
  on mixed recordings not more than 0.5 points worse; (c) FLEURS Russian, FLEURS English and Golos each not more
  than 0.5 points worse.
* Rule 2 (routes): each language to the installed recognizer with the lowest pooled validation WER; the recognizer
  for the mixed setting by the round-2 rule (lowest one-pass WER on mixed recordings, a tie to the smaller model).
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VALIDATION_SETS = ("mixed-val", "fleurs-val", "golos-val", "ksc2-val")
TOLERANCE, RELATIVE_GAIN = 0.005, 0.10
SIZE = {"small": 1, "cleartalk": 1, "cleartalk-kk": 1, "turbo": 2, "kazakh": 2, "cleartalk-large": 2}
EXPORT_NAME = "whisper-small-cleartalk-kazakh"


def main():
    import pandas as pd

    from backend.config import DEFAULT_MULTILINGUAL_PROFILE
    from cleartalk_ml import analysis, validation
    from cleartalk_ml.awake import keep_awake
    from cleartalk_ml.convert import sha256
    from cleartalk_ml.evaluate import bootstrap_difference, details, hyp_path, pick_device, run_units, units_of
    from cleartalk_ml.export import export
    from cleartalk_ml.paths import MODELS, RESULTS, RUNS
    from cleartalk_ml.text import pooled

    keep_awake()
    target = MODELS / EXPORT_NAME
    choice = json.loads((RUNS / "kk-lr-choice.json").read_text(encoding="utf-8"))
    run = choice["chosen"]
    run_dir = RUNS / run
    if not (run_dir / "finished.json").is_file():
        raise SystemExit("Training has not finished; run scripts/run_training.py --data kazakh-broadcast --rate 5e-4 first")
    finished = json.loads((run_dir / "finished.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    steps = sorted(int(p.name.split("-")[1]) for p in run_dir.glob("step-*") if (p / "adapter_model.safetensors").is_file())
    shortlist = sorted({steps[-1], int(finished["best"]["step"])} & set(steps)) or [steps[-1]]
    device, precision = pick_device()
    round1 = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    round2 = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
    old = f"runs/{round1['run']}/ct2-step-{round1['chosen_step']:05d}"
    large = f"runs/{round2['run']}/ct2-step-{round2['chosen_step']:05d}"

    def system_of(step):
        return f"runs/{run}/ct2-step-{step:05d}"

    def score(system, sets=VALIDATION_SETS):
        for set_name in sets:
            run_units(system, units_of(set_name), hyp_path(set_name, system, device, precision), device, precision)

    def kazakh_rows(system):
        rows = []
        for set_name in ("fleurs-val", "ksc2-val"):
            group = units_of(set_name, language="kk")
            hyp, _ = validation.hypotheses(set_name, system)
            rows.append(details(group, hyp, "forced"))
        return pd.concat(rows, ignore_index=True)

    def judge(system):
        single, base_single = validation.single_scores(system), validation.single_scores(old)
        mixed, base_mixed = validation.mixed_scores(system), validation.mixed_scores(old)
        kk = bootstrap_difference(kazakh_rows(old), kazakh_rows(system))
        base_kk = pooled(kazakh_rows(old).to_dict("records"))["wer"]
        new_kk = pooled(kazakh_rows(system).to_dict("records"))["wer"]
        guards = {key: {"round1_wer": base_single[key]["forced"]["wer"], "candidate_wer": single[key]["forced"]["wer"],
                        "worse_by": single[key]["forced"]["wer"] - base_single[key]["forced"]["wer"]}
                  for key in ("fleurs-ru", "fleurs-en", "golos-ru")}
        guards["mixed"] = {"round1_wer": base_mixed["wer"], "candidate_wer": mixed["wer"], "worse_by": mixed["wer"] - base_mixed["wer"]}
        for block in guards.values():
            block["ok"] = block["worse_by"] <= TOLERANCE
        relative = (new_kk - base_kk) / base_kk
        rule_a = relative <= -RELATIVE_GAIN and kk["high"] < 0
        return {"system": system, "kazakh": {"round1_wer": base_kk, "candidate_wer": new_kk, "relative_change": relative, **kk},
                "guards": guards, "rule_a_kazakh_gain": bool(rule_a), "rule_b_mixed_ok": guards["mixed"]["ok"],
                "rule_c_languages_ok": all(guards[k]["ok"] for k in ("fleurs-ru", "fleurs-en", "golos-ru")),
                "promoted": bool(rule_a and guards["mixed"]["ok"] and all(guards[k]["ok"] for k in ("fleurs-ru", "fleurs-en", "golos-ru")))}

    score(old)                                     # cached since round 1
    candidates = {}
    for step in shortlist:
        output = MODELS / system_of(step)
        if not (output / "model_card.json").is_file():
            export(run_dir / f"step-{step:05d}", output, {"step": step}, base="small")
        score(system_of(step))
        candidates[step] = judge(system_of(step))
        print(f"step {step}: Kazakh {100 * candidates[step]['kazakh']['candidate_wer']:.2f}% (round 1 "
              f"{100 * candidates[step]['kazakh']['round1_wer']:.2f}%), guards ok: {candidates[step]['rule_b_mixed_ok'] and candidates[step]['rule_c_languages_ok']}", flush=True)
    meeting = [step for step, item in candidates.items() if item["rule_b_mixed_ok"] and item["rule_c_languages_ok"]]
    pool = meeting or list(candidates)
    step = min(pool, key=lambda item: (round(candidates[item]["kazakh"]["candidate_wer"], 5), -item))
    decision, system = candidates[step], system_of(step)

    # The app's model: a copy of the very folder that was validated, plus its card.
    percent = lambda value: f"{100 * value:.1f}%"  # noqa: E731
    single, base_single = validation.single_scores(system), validation.single_scores(old)
    mixed, base_mixed = validation.mixed_scores(system), validation.mixed_scores(old)
    sources = {"round1": analysis.ksc2_by_source(old), "candidate": analysis.ksc2_by_source(system)}
    labels = (("fleurs-kk", "Kazakh read sentences"), ("ksc2-kk", "Kazakh broadcast and read speech"), ("fleurs-ru", "Russian read sentences"),
              ("golos-ru", "short Russian phrases"), ("fleurs-en", "English read sentences"))
    evidence = [
        "ClearTalk multilingual with Kazakh broadcast: Whisper small adapted with LoRA to Kazakh, Russian, English, to recordings "
        "that change language and to Kazakh radio, podcasts, talk shows, parliament and TV news.",
        "6,488,064 trained LoRA parameters (2.6% of 248,222,976); merged, so it runs as fast as Whisper small.",
        f"Trained on {config['train_examples']:,} examples ({config['train_mixed']:,} of them mixed-language recordings), one pass, on a GTX 1660 Ti.",
        f"Validation, mixed-language recordings ({mixed['units']}): word error rate {percent(base_mixed['wer'])} ClearTalk small → "
        f"{percent(mixed['wer'])} ClearTalk small (Kazakh broadcast). Lower is better.",
    ]
    for key, label in labels:
        evidence.append(f"Validation, {label} ({single[key]['units']}): {percent(base_single[key]['forced']['wer'])} ClearTalk small → "
                        f"{percent(single[key]['forced']['wer'])} ClearTalk small (Kazakh broadcast).")
    evidence.append("Mixed recordings are assembled from single-language utterances: every language change is also a change of "
                    "speaker. Real conversations, accents and microphones can behave differently.")
    card = {"model_name": "ClearTalk multilingual Whisper small with Kazakh broadcast speech (LoRA fine-tuned)", "run": run, "step": step,
            "trainable_parameters": 6488064, "total_parameters": 248222976,
            "training": {"examples": config["train_examples"], "mixed_recordings": config["train_mixed"], "learning_rate": config["lr"],
                         "optimizer_steps": finished["last_step"], "composition": config.get("data")},
            "baseline": "the ClearTalk small model of round 1", "promotion": decision, "evidence": evidence}
    if target.is_dir():
        shutil.rmtree(target)
    shutil.copytree(MODELS / system, target)
    card.update(adapter=f"models/multilingual/runs/{run}/step-{step:05d}", base_model="openai/whisper-small",
                files_sha256={p.name: sha256(p) for p in sorted(target.iterdir()) if p.is_file() and p.name != "model_card.json"})
    (target / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    same = sha256(target / "model.bin") == sha256(MODELS / system / "model.bin")
    print(f"exported step {step}; identical to the validated file: {same}", flush=True)
    if not same:
        raise SystemExit("The exported model differs from the validated checkpoint")

    # Rule 2 and the round-2 rule for the mixed setting, with every recognizer scored on every set of its languages.
    for reference in ("turbo", "small", large):
        score(reference)
    score("kazakh")
    systems = {"turbo": "turbo", "kazakh": "kazakh", "small": "small", "cleartalk": old, "cleartalk-large": large, "cleartalk-kk": system}
    profile = json.loads(json.dumps(DEFAULT_MULTILINGUAL_PROFILE))
    covers = {key: entry["languages"] for key, entry in profile["recognizers"].items()}
    routing = validation.routes(systems, covers)
    for language, key in routing["routes"].items():
        profile["routes"][language] = key
    one_pass = {key: validation.mixed_scores(name)["wer"] for key, name in systems.items() if set(covers[key]) == {"kk", "ru", "en"}}
    profile["mixed_recognizer"] = min(one_pass, key=lambda key: (round(one_pass[key], 5), SIZE[key], key))
    profile["small_recognizer"] = "cleartalk-kk" if decision["promoted"] else "cleartalk"
    for key, entry in profile["recognizers"].items():
        directory = MODELS / entry["directory"]
        if (directory / "model.bin").is_file():
            entry["files_sha256"] = {name: sha256(directory / name) for name in ("model.bin", "config.json", "tokenizer.json")}
    (MODELS / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {"run": run, "checkpoints": {str(k): v for k, v in candidates.items()}, "chosen_step": step, "exported_identical": same,
               "rule_1_promotion": decision, "rule_2_routes": routing, "mixed_recognizer": {"chosen": profile["mixed_recognizer"], "one_pass": one_pass},
               "small_recognizer": profile["small_recognizer"], "ksc2_by_source": sources}
    (RESULTS / "selection-round3.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"chosen_step": step, "promoted": decision["promoted"], "routes": routing["routes"],
                      "mixed_recognizer": profile["mixed_recognizer"], "small_recognizer": profile["small_recognizer"]}, indent=1))
    print("SELECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
