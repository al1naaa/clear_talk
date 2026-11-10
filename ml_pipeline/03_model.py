import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(ROOT))
    from cleartalk_ml import pipeline_log as log
    from cleartalk_ml import validation
    from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES, RESULTS, RUNS

    COLORS = {"kk": "#2a9d8f", "ru": "#3b6ea8", "en": "#d08c2b"}
    SYSTEM_NAMES = {"small": "Whisper small\n(untouched)", "turbo": "Whisper\nlarge-v3-turbo", "kazakh": "Kazakh\nturbo fine-tune",
                    "pipeline:auto": "App pipeline\n(ready models)", "cleartalk": "ClearTalk\n(ours)"}

    def read_history(run):
        path = RUNS / run / "history.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []

    return COLORS, LANGUAGES, LANGUAGE_NAMES, RESULTS, RUNS, SYSTEM_NAMES, json, log, mo, np, pd, plt, read_history, validation


@app.cell
def _(mo):
    mo.md("""
    # Steps 9-12 · Baselines, training, tuning and validation (multilingual model)

    Everything here is measured on **validation** data. The test units stay frozen and unopened.
    """)
    return


@app.cell
def _(COLORS, LANGUAGES, LANGUAGE_NAMES, SYSTEM_NAMES, log, np, plt, validation):
    # Step 9 - baselines through the app's decoding path.
    baseline_systems = ["small", "turbo", "kazakh"]
    singles = {system: validation.single_scores(system) for system in baseline_systems}
    mixed = {system: validation.mixed_scores(system) for system in ("small", "turbo", "pipeline:auto")}
    mixed = {system: block for system, block in mixed.items() if block is not None}
    _f, _axes = plt.subplots(1, 3, figsize=(16, 4))
    _sets = ["fleurs-kk", "fleurs-ru", "fleurs-en", "ksc2-kk", "golos-ru"]
    _x = np.arange(len(_sets))
    for _i, _system in enumerate(baseline_systems):
        _vals = [100 * singles[_system][s]["forced"]["wer"] if s in singles[_system] else np.nan for s in _sets]
        _axes[0].bar(_x + (_i - 1) * 0.27, _vals, 0.27, label=SYSTEM_NAMES[_system].replace("\n", " "))
    _axes[0].set_xticks(_x, _sets)
    _axes[0].set_ylabel("WER %, language given")
    _axes[0].set_title("Single-language utterances (validation)")
    _axes[0].legend(fontsize=8)
    _names = list(mixed)
    _axes[1].bar([SYSTEM_NAMES[n] for n in _names], [100 * mixed[n]["wer"] for n in _names], color="#de693c")
    _axes[1].axhline(100, color="black", linewidth=0.8, linestyle=":")
    _axes[1].set_ylabel("WER %, automatic language")
    _axes[1].set_title("Mixed-language recordings (validation); dotted = empty transcript")
    _axes[1].tick_params(axis="x", labelsize=8)
    for _i, _lang in enumerate(LANGUAGES):
        _axes[2].bar(np.arange(len(_names)) + (_i - 1) * 0.27, [100 * mixed[n]["by_language"][_lang]["wer"] for n in _names], 0.27,
                     color=COLORS[_lang], label=LANGUAGE_NAMES[_lang])
    _axes[2].set_xticks(np.arange(len(_names)), [SYSTEM_NAMES[n] for n in _names], fontsize=8)
    _axes[2].set_ylabel("WER % of the words of each language")
    _axes[2].set_title("Inside mixed recordings: errors by language")
    _axes[2].legend(fontsize=8)
    _small = singles["small"]
    log.fig(9, "baselines", _f,
            "An empty transcript scores 100% WER. Untouched Whisper small through the app's decoding: "
            + ", ".join(f"{s} {100 * _small[s]['forced']['wer']:.1f}%" for s in _sets if s in _small)
            + (f"; mixed recordings {100 * mixed['small']['wer']:.1f}% with automatic language" if "small" in mixed else "")
            + ". Reference points: "
            + "; ".join(f"{name.replace(chr(10), ' ')} " + ", ".join(f"{s} {100 * singles[key][s]['forced']['wer']:.1f}%" for s in _sets if s in singles[key])
                        for key, name in (("turbo", SYSTEM_NAMES["turbo"]), ("kazakh", SYSTEM_NAMES["kazakh"])))
            + ("; mixed recordings: " + ", ".join(f"{SYSTEM_NAMES[n].replace(chr(10), ' ')} {100 * mixed[n]['wer']:.1f}%" for n in _names if n != "small")
               if len(_names) > 1 else "")
            + ". The small model is far behind on Kazakh and on mixed recordings: that gap is what training targets.")
    {"single": {system: {s: round(100 * block["forced"]["wer"], 2) for s, block in scores.items()} for system, scores in singles.items()},
     "mixed": {system: round(100 * block["wer"], 2) for system, block in mixed.items()}}
    return mixed, singles


@app.cell
def _(RUNS, json, log, plt, read_history):
    # Step 11 - learning-rate pilots: identical data, seed and schedule; only the rate differs.
    choice = json.loads((RUNS / "lr-choice.json").read_text(encoding="utf-8"))
    _f, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for _run, _info in choice["pilots"].items():
        _train = [r for r in read_history(_run) if r.get("type") == "train" and r["step"] <= _info["step"]]
        _axes[0].plot([r["step"] for r in _train], [r["loss"] for r in _train], label=f"{_info['lr']:g}")
    _axes[0].set_xlabel("optimizer step")
    _axes[0].set_ylabel("training loss")
    _axes[0].set_title("Pilots: training loss")
    _axes[0].legend(title="learning rate")
    _labels = [f"{info['lr']:g}" for info in choice["pilots"].values()]
    _axes[1].bar(_labels, [100 * info["val_mixed_wer"] for info in choice["pilots"].values()], color="#de693c")
    _axes[1].set_ylabel("validation WER % on mixed recordings (greedy subset)")
    _axes[1].set_title(f"After {next(iter(choice['pilots'].values()))['step']} steps")
    log.fig(11, "learning_rate_pilots", _f,
            f"Identical {next(iter(choice['pilots'].values()))['step']}-step runs that differ only in learning rate, compared on a "
            "fixed validation subset of mixed recordings: "
            + ", ".join(f"{info['lr']:g} → {100 * info['val_mixed_wer']:.1f}%" for info in choice["pilots"].values())
            + f". {choice['chosen']} was continued as the full run; the test sets played no part.")
    choice
    return (choice,)


@app.cell
def _(LANGUAGES, choice, log, np, plt, read_history):
    # Step 10 - learning curves of the main run: training and validation side by side.
    history = read_history(choice["chosen"])
    evals = [r for r in history if r.get("type") == "eval"]
    seen = {}
    for _r in evals:
        seen[_r["step"]] = _r  # a resumed run may log a step twice; keep the last
    evals = [seen[k] for k in sorted(seen)]
    train_log = [r for r in history if r.get("type") == "train"]
    _steps = [r["step"] for r in evals]
    _f, _axes = plt.subplots(1, 3, figsize=(16, 4))
    _axes[0].plot(_steps, [100 * r["val_mixed"]["wer"] for r in evals], "o-", color="#de693c", label="validation, mixed")
    _axes[0].plot(_steps, [100 * r["train_mixed"]["wer"] for r in evals], "o--", color="#777777", label="training subset, mixed")
    _axes[0].set_xlabel("optimizer step (0 = untouched Whisper small)")
    _axes[0].set_ylabel("WER %, greedy decoding")
    _axes[0].set_title("Mixed recordings: training vs validation")
    _axes[0].legend(fontsize=8)
    for _key, _label in ([(f"val_fleurs-{lang}", f"FLEURS {lang}") for lang in LANGUAGES] + [("val_ksc2-kk", "KSC2 kk"), ("val_golos-ru", "Golos ru")]):
        _axes[1].plot(_steps, [100 * r[_key]["wer"] for r in evals], "o-", label=_label)
    _axes[1].set_xlabel("optimizer step")
    _axes[1].set_ylabel("WER %, greedy decoding")
    _axes[1].set_title("Single-language validation sets")
    _axes[1].legend(fontsize=8)
    _axes[2].plot([r["step"] for r in train_log], [r["loss"] for r in train_log], color="#bbbbbb", label="training loss (running)")
    _axes[2].plot(_steps, [r["val_loss"] for r in evals], "o-", color="#de693c", label="validation loss, mixed")
    _axes[2].plot(_steps, [r["train_loss"] for r in evals], "o--", color="#777777", label="training-subset loss, mixed")
    _axes[2].set_xlabel("optimizer step")
    _axes[2].set_ylabel("cross-entropy per token")
    _axes[2].set_title("Loss")
    _axes[2].legend(fontsize=8)
    first, last = evals[0], evals[-1]
    _rel = lambda key: 100 * (last[key]["wer"] / first[key]["wer"] - 1)  # noqa: E731
    log.fig(10, "learning_curves", _f,
            f"Greedy decoding on fixed subsets. Mixed recordings: validation WER {100 * first['val_mixed']['wer']:.1f}% (step 0 = untouched "
            f"Whisper small) → {100 * last['val_mixed']['wer']:.1f}% at step {last['step']}; the training subset "
            f"{100 * first['train_mixed']['wer']:.1f}% → {100 * last['train_mixed']['wer']:.1f}%. Relative change: training {_rel('train_mixed'):+.0f}% "
            f"vs validation {_rel('val_mixed'):+.0f}%; validation loss {first['val_loss']:.3f} → {last['val_loss']:.3f} "
            f"(training-subset loss {first['train_loss']:.3f} → {last['train_loss']:.3f}). "
            "Single-language sets: " + ", ".join(f"{k.replace('val_', '')} {100 * first[k]['wer']:.1f}% → {100 * last[k]['wer']:.1f}%"
                                                 for k in sorted(first) if k.startswith("val_") and k not in ("val_mixed", "val_loss"))
            + ". A training curve that keeps falling while the validation curve turns up would mean overfitting.")
    {"steps": _steps, "val_mixed": [round(100 * r["val_mixed"]["wer"], 2) for r in evals]}
    return


@app.cell
def _(COLORS, LANGUAGES, LANGUAGE_NAMES, RESULTS, SYSTEM_NAMES, json, log, mixed, np, plt, singles, validation):
    # Step 12 - the chosen checkpoint through the app's decoding, against the baselines, on validation.
    selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    system = f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"
    ours_single, ours_mixed = validation.single_scores(system), validation.mixed_scores(system)
    promo = selection["promotion"]
    _sets = ["fleurs-kk", "fleurs-ru", "fleurs-en", "ksc2-kk", "golos-ru"]
    _f, _axes = plt.subplots(1, 3, figsize=(17, 4.2))
    _systems = {"small": singles["small"], "cleartalk": ours_single, "turbo": singles["turbo"], "kazakh": singles["kazakh"]}
    _x = np.arange(len(_sets))
    for _i, (_key, _scores) in enumerate(_systems.items()):
        _axes[0].bar(_x + (_i - 1.5) * 0.2, [100 * _scores[s]["forced"]["wer"] if s in _scores else np.nan for s in _sets], 0.2,
                     label=SYSTEM_NAMES[_key].replace("\n", " "), color={"small": "#999999", "cleartalk": "#de693c", "turbo": "#3b6ea8", "kazakh": "#2a9d8f"}[_key])
    _axes[0].set_xticks(_x, _sets)
    _axes[0].set_ylabel("WER %, language given")
    _axes[0].set_title("Single-language utterances (validation, app decoding)")
    _axes[0].legend(fontsize=8)
    _mixed = {"small": mixed.get("small"), "cleartalk": ours_mixed, "turbo": mixed.get("turbo"), "pipeline:auto": mixed.get("pipeline:auto")}
    _mixed = {k: v for k, v in _mixed.items() if v is not None}
    _axes[1].bar([SYSTEM_NAMES[k] for k in _mixed], [100 * v["wer"] for v in _mixed.values()],
                 color=["#de693c" if k == "cleartalk" else "#999999" for k in _mixed])
    _axes[1].set_ylabel("WER %, automatic language")
    _axes[1].set_title("Mixed-language recordings (validation)")
    _axes[1].tick_params(axis="x", labelsize=8)
    for _i, _lang in enumerate(LANGUAGES):
        _axes[2].bar(np.arange(len(_mixed)) + (_i - 1) * 0.27, [100 * v["by_language"][_lang]["wer"] for v in _mixed.values()], 0.27,
                     color=COLORS[_lang], label=LANGUAGE_NAMES[_lang])
    _axes[2].set_xticks(np.arange(len(_mixed)), [SYSTEM_NAMES[k] for k in _mixed], fontsize=8)
    _axes[2].set_ylabel("WER % of the words of each language")
    _axes[2].set_title("Inside mixed recordings: errors by language")
    _axes[2].legend(fontsize=8)
    _m = promo["mixed"]
    log.fig(12, "validation_comparison", _f,
            f"Through the app's exact decoding, validation WER on mixed recordings is {100 * _m['baseline_wer']:.1f}% (Whisper small) → "
            f"{100 * _m['candidate_wer']:.1f}% (ours, step {selection['chosen_step']}): difference {100 * _m['difference']:+.1f} points, 95% bootstrap "
            f"interval {100 * _m['low']:+.1f} to {100 * _m['high']:+.1f} over {_m['recordings']} recordings ({100 * _m['relative_change']:+.0f}% relative). "
            "FLEURS with the language given: "
            + ", ".join(f"{LANGUAGE_NAMES[lang]} {100 * promo['languages'][lang]['baseline_wer']:.1f}% → {100 * promo['languages'][lang]['candidate_wer']:.1f}%"
                        for lang in LANGUAGES)
            + "; other domains: " + ", ".join(f"{s} {100 * singles['small'][s]['forced']['wer']:.1f}% → {100 * ours_single[s]['forced']['wer']:.1f}%"
                                              for s in ("ksc2-kk", "golos-ru") if s in ours_single and s in singles["small"])
            + ". Reference points on mixed recordings: "
            + ", ".join(f"{SYSTEM_NAMES[k].replace(chr(10), ' ')} {100 * v['wer']:.1f}%" for k, v in _mixed.items() if k not in ("small", "cleartalk"))
            + f". Promotion rule: mixed gain {'met' if promo['rule_a_mixed_gain'] else 'not met'}, no language worse "
            f"{'met' if promo['rule_b_no_language_worse'] else 'not met'} → {'promoted' if promo['promoted'] else 'not promoted'}. "
            f"Routes chosen by validation: {selection['routes']['routes']}.")
    selection
    return


if __name__ == "__main__":
    app.run()
