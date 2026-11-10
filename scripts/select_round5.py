"""Round 5: choose the checkpoint of the large model trained in noise WITH the top encoder layers, and apply the rule.

    python scripts/select_round5.py

Rules (ml_pipeline/round5/PIPELINE.md, written before training):
1. Checkpoint - of the saved steps, the lowest pooled one-pass WER over the noisy robustness validation
   recordings among those that meet the clean guard of rule 2(b); none meets it -> the round-2 model stays.
2. Promotion - (a) that pooled noisy WER is at least 5% lower, relative, than the round-2 model's, with the 95%
   bootstrap interval of the difference below zero; (b) clean validation sets (mixed recordings one pass; FLEURS
   kk/ru/en, KSC2, Golos with the language given) each not more than 0.5 points worse.
Promoted -> the checkpoint folder is copied as models/multilingual/whisper-large-cleartalk-multilingual (the round-2
model stays in models/multilingual/runs/) and the profile's signatures are updated. Transcripts are cached.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NOISY = ("white-20", "white-10", "babble-10", "babble-5")
CLEAN_SINGLE = (("fleurs-val", "fleurs-kk"), ("fleurs-val", "fleurs-ru"), ("fleurs-val", "fleurs-en"), ("ksc2-val", "ksc2-kk"), ("golos-val", "golos-ru"))
TOLERANCE, GAIN = 0.005, 0.05


def main():
    import pandas as pd

    from cleartalk_ml import robustness, validation
    from cleartalk_ml.awake import keep_awake
    from cleartalk_ml.convert import sha256
    from cleartalk_ml.evaluate import bootstrap_difference, details, hyp_path, pick_device, read_hypotheses, run_units, units_of
    from cleartalk_ml.export import export
    from cleartalk_ml.paths import BASES, MODELS, RESULTS, RUNS
    from cleartalk_ml.text import pooled

    keep_awake()
    device, precision = pick_device()
    backend = f"{device}-{precision}"
    choice = json.loads((RUNS / "large-noise-enc8-lr-choice.json").read_text(encoding="utf-8"))
    run = choice["chosen"]
    run_dir = RUNS / run
    if not (run_dir / "finished.json").is_file():
        raise SystemExit("Training has not finished; run scripts/run_training.py --base kazakh-turbo --noise --encoder 8 --rate 1e-4 first")
    steps = sorted(int(p.name.split("-")[1]) for p in run_dir.glob("step-*") if (p / "adapter_model.safetensors").is_file() and p.name != "step-00000")
    round2 = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
    baseline = f"runs/{round2['run']}/ct2-step-{round2['chosen_step']:05d}"
    plan = json.loads(robustness.PLAN.read_text(encoding="utf-8"))

    def system_of(step):
        return f"runs/{run}/ct2-step-{step:05d}"

    def noisy_frame(system):
        frames = []
        for condition in NOISY:
            units = robustness.units(plan, condition)
            units = units[units.set != "long"].reset_index(drop=True)
            path = robustness.hyp_file(condition, system.replace("/", "_"), backend)
            run_units(system, units, path, device, precision)
            hyp = read_hypotheses(path)
            for set_name, group in units.groupby("set"):
                frames.append(details(group, hyp, "auto" if set_name == "mixed" else "forced").assign(unit_id=lambda f, c=condition: c + ":" + f.unit_id))
        return pd.concat(frames, ignore_index=True)

    def clean_scores(system):
        run_units(system, units_of("mixed-val"), hyp_path("mixed-val", system, device, precision), device, precision)
        for set_name in ("fleurs-val", "ksc2-val", "golos-val"):
            run_units(system, units_of(set_name), hyp_path(set_name, system, device, precision), device, precision, forced_only=True)
        scores = {"mixed": validation.mixed_scores(system)["wer"]}
        for set_name, key in CLEAN_SINGLE:
            hyp, _ = validation.hypotheses(set_name, system)
            scores[key] = pooled(details(units_of(set_name, language=key.split("-")[1]), hyp, "forced").to_dict("records"))["wer"]
        return scores

    wer = lambda frame: pooled(frame.to_dict("records"))["wer"]  # noqa: E731
    base_noisy = noisy_frame(baseline)
    base_clean = clean_scores(baseline)
    candidates = {}
    for step in steps:
        output = MODELS / system_of(step)
        if not (output / "model_card.json").is_file():
            export(run_dir / f"step-{step:05d}", output, {"step": step}, base="kazakh-turbo")
        frame = noisy_frame(system_of(step))
        candidates[step] = {"noisy_wer": wer(frame), "frame": frame}
        print(f"step {step}: noisy validation WER {100 * candidates[step]['noisy_wer']:.2f}% (round 2: {100 * wer(base_noisy):.2f}%)", flush=True)
    chosen, guard = None, {}
    for step in sorted(candidates, key=lambda s: (round(candidates[s]["noisy_wer"], 5), s)):
        scores = clean_scores(system_of(step))
        guard[step] = {key: {"round2": base_clean[key], "candidate": value, "worse_by": value - base_clean[key],
                             "ok": value - base_clean[key] <= TOLERANCE} for key, value in scores.items()}
        if all(item["ok"] for item in guard[step].values()):
            chosen = step
            break
    result = {"run": run, "baseline": baseline, "baseline_noisy_wer": wer(base_noisy), "baseline_clean": base_clean,
              "checkpoints_noisy_wer": {str(s): c["noisy_wer"] for s, c in candidates.items()},
              "clean_guard": {str(s): g for s, g in guard.items()}, "chosen_step": chosen}
    if chosen is not None:
        interval = bootstrap_difference(base_noisy, candidates[chosen]["frame"])
        relative = (candidates[chosen]["noisy_wer"] - wer(base_noisy)) / wer(base_noisy)
        by_condition = {}
        for condition in NOISY:
            a = base_noisy[base_noisy.unit_id.str.startswith(condition + ":")]
            b = candidates[chosen]["frame"][candidates[chosen]["frame"].unit_id.str.startswith(condition + ":")]
            by_condition[condition] = {"round2": wer(a), "candidate": wer(b)}
        promoted = relative <= -GAIN and interval["high"] < 0
        result.update(noisy={"relative_change": relative, **interval}, by_condition=by_condition, promoted=bool(promoted))
        if promoted:
            target = MODELS / BASES["kazakh-turbo"]["export"]
            card = json.loads((target / "model_card.json").read_text(encoding="utf-8"))
            shutil.rmtree(target)
            shutil.copytree(MODELS / system_of(chosen), target)
            card.update(run=run, step=chosen, adapter=f"models/multilingual/runs/{run}/step-{chosen:05d}", round=5,
                        noise_training={"examples_with_noise": 0.5, "warm_start": baseline, "learning_rate": 1e-4, "encoder_layers": 8},
                        files_sha256={p.name: sha256(p) for p in sorted(target.iterdir()) if p.is_file() and p.name != "model_card.json"})
            (target / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            profile_path = MODELS / "profile.json"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["recognizers"]["cleartalk-large"]["files_sha256"] = {name: sha256(target / name) for name in ("model.bin", "config.json", "tokenizer.json")}
            profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            result["exported_identical"] = sha256(target / "model.bin") == sha256(MODELS / system_of(chosen) / "model.bin")
    else:
        result["promoted"] = False
    (RESULTS / "selection-round5.json").write_text(json.dumps({k: v for k, v in result.items()}, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("chosen_step", "promoted") if k in result}, indent=1), flush=True)
    print("SELECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
