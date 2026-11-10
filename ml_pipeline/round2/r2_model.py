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

    ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(ROOT))
    from cleartalk_ml import pipeline_log as log
    from cleartalk_ml import round2, validation
    from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES, RESULTS, RUNS

    log.use(round2.R2)
    COLORS = {"kk": "#2a9d8f", "ru": "#3b6ea8", "en": "#d08c2b"}
    SETS = ["fleurs-kk", "ksc2-kk", "fleurs-ru", "golos-ru", "fleurs-en"]
    round1 = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    SMALL_OURS = f"runs/{round1['run']}/ct2-step-{round1['chosen_step']:05d}"
    # name on the figures -> (system in the transcript files, colour)
    SYSTEMS = {"Kazakh turbo\n(starting model)": ("kazakh", "#2a9d8f"), "large-v3-turbo\n(untouched)": ("turbo", "#3b6ea8"),
               "Whisper small\n(untouched)": ("small", "#999999"), "Our small model\n(round 1)": (SMALL_OURS, "#e9b384")}

    def read_history(run):
        path = RUNS / run / "history.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []

    def flat(name):
        return name.replace("\n", " ")

    return COLORS, LANGUAGES, LANGUAGE_NAMES, RESULTS, RUNS, SETS, SYSTEMS, flat, json, log, mo, np, plt, read_history, validation


@app.cell
def _(mo):
    mo.md("""
    # Round 2 · Steps 9-12: baselines, training, tuning and validation of the large model

    Everything here is measured on **validation** data, through the app's decoding unless a figure says
    "greedy". Test B stays frozen and unopened.
    """)
    return


@app.cell
def _(COLORS, LANGUAGES, LANGUAGE_NAMES, SETS, SYSTEMS, flat, log, np, plt, validation):
    # Step 9 - baselines. The starting model untouched is THE baseline of this round; the rest are reference points.
    singles = {name: validation.single_scores(system) for name, (system, _) in SYSTEMS.items()}
    mixed = {name: validation.mixed_scores(system) for name, (system, _) in SYSTEMS.items()}
    mixed["App pipeline\n(round-1 routes)"] = validation.mixed_scores("pipeline:auto")
    mixed = {name: block for name, block in mixed.items() if block is not None}
    _colors = {**{name: color for name, (_, color) in SYSTEMS.items()}, "App pipeline\n(round-1 routes)": "#555555"}
    _f, _axes = plt.subplots(1, 3, figsize=(18, 4.3), gridspec_kw={"width_ratios": [2.2, 1.3, 1.6]})
    _x = np.arange(len(SETS))
    for _i, _name in enumerate(SYSTEMS):
        _vals = [min(110, 100 * singles[_name][s]["forced"]["wer"]) if s in singles[_name] else np.nan for s in SETS]
        _axes[0].bar(_x + (_i - 1.5) * 0.2, _vals, 0.2, color=_colors[_name], label=flat(_name))
    _axes[0].set_xticks(_x, SETS)
    _axes[0].set_ylabel("WER %, language given (cut at 110)")
    _axes[0].set_title("Single-language utterances (validation)")
    _axes[0].legend(fontsize=8)
    _axes[1].bar(list(mixed), [100 * block["wer"] for block in mixed.values()], color=[_colors[name] for name in mixed])
    _axes[1].axhline(100, color="black", linewidth=0.8, linestyle=":")
    _axes[1].set_ylabel("WER %, automatic language")
    _axes[1].set_title("Mixed recordings (validation); dotted = empty transcript")
    _axes[1].tick_params(axis="x", labelsize=6.5)
    for _i, _lang in enumerate(LANGUAGES):
        _axes[2].bar(np.arange(len(mixed)) + (_i - 1) * 0.27, [min(130, 100 * block["by_language"][_lang]["wer"]) for block in mixed.values()], 0.27,
                     color=COLORS[_lang], label=LANGUAGE_NAMES[_lang])
    _axes[2].set_xticks(np.arange(len(mixed)), list(mixed), fontsize=6.5)
    _axes[2].set_ylabel("WER % of the words of each language (cut at 130)")
    _axes[2].set_title("Inside mixed recordings: errors by language")
    _axes[2].legend(fontsize=8)
    _f.tight_layout()
    _start, _start_mixed = singles["Kazakh turbo\n(starting model)"], mixed["Kazakh turbo\n(starting model)"]
    log.fig(9, "baselines", _f,
            "An empty transcript scores 100% WER. The baseline of this round is the starting model untouched (the Kazakh turbo "
            "fine-tune) through the app's decoding: "
            + ", ".join(f"{s} {100 * _start[s]['forced']['wer']:.1f}%" for s in SETS if s in _start)
            + f"; mixed recordings in one pass {100 * _start_mixed['wer']:.1f}% (words of each language inside them: "
            + ", ".join(f"{LANGUAGE_NAMES[lang]} {100 * _start_mixed['by_language'][lang]['wer']:.0f}%" for lang in LANGUAGES)
            + "). Reference points on mixed recordings: "
            + ", ".join(f"{flat(name)} {100 * block['wer']:.1f}%" for name, block in mixed.items() if not name.startswith("Kazakh"))
            + ". The starting model hears Kazakh well but, in a recording that changes language, keeps writing in the language "
            "it started with: that habit belongs to the decoder, which is the part this round trains.")
    {"single": {flat(name): {s: round(100 * block["forced"]["wer"], 2) for s, block in scores.items()} for name, scores in singles.items()},
     "mixed": {flat(name): round(100 * block["wer"], 2) for name, block in mixed.items()}}
    return mixed, singles


@app.cell
def _(RUNS, json, log, plt, read_history):
    # Step 11 - learning-rate pilots: identical data, seed and schedule; only the rate differs.
    choice = json.loads((RUNS / "large-lr-choice.json").read_text(encoding="utf-8"))
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
    _f.tight_layout()
    log.fig(11, "learning_rate_pilots", _f,
            f"Identical {next(iter(choice['pilots'].values()))['step']}-step runs of the large model that differ only in learning rate, "
            "compared on a fixed validation subset of mixed recordings (greedy decoding): "
            + ", ".join(f"{info['lr']:g} → {100 * info['val_mixed_wer']:.1f}%" for info in choice["pilots"].values())
            + "; FLEURS kk/ru/en after the pilot: "
            + "; ".join(f"{info['lr']:g}: " + "/".join(f"{100 * info['val_fleurs_wer'][lang]:.1f}" for lang in ("kk", "ru", "en")) for info in choice["pilots"].values())
            + f". {choice['chosen']} was continued as the full run; test B played no part.")
    choice
    return (choice,)


@app.cell
def _(LANGUAGES, choice, log, plt, read_history):
    # Step 10 - learning curves of the main run: training and validation side by side.
    history = read_history(choice["chosen"])
    seen = {}
    for _r in history:
        if _r.get("type") == "eval":
            seen[_r["step"]] = _r  # a resumed run may log a step twice; keep the last
    evals = [seen[k] for k in sorted(seen)]
    train_log = [r for r in history if r.get("type") == "train"]
    _steps = [r["step"] for r in evals]
    _f, _axes = plt.subplots(1, 3, figsize=(16, 4))
    _axes[0].plot(_steps, [100 * r["val_mixed"]["wer"] for r in evals], "o-", color="#de693c", label="validation, mixed")
    _axes[0].plot(_steps, [100 * r["train_mixed"]["wer"] for r in evals], "o--", color="#777777", label="training subset, mixed")
    _axes[0].set_xlabel("optimizer step (0 = the untouched starting model)")
    _axes[0].set_ylabel("WER %, greedy decoding")
    _axes[0].set_title("Mixed recordings: training vs validation")
    _axes[0].legend(fontsize=8)
    for _key, _label in ([(f"val_fleurs-{lang}", f"FLEURS {lang}") for lang in LANGUAGES] + [("val_ksc2-kk", "KSC2 kk"), ("val_golos-ru", "Golos ru")]):
        _axes[1].plot(_steps, [100 * r[_key]["wer"] for r in evals], "o-", label=_label)
    _axes[1].set_xlabel("optimizer step")
    _axes[1].set_ylabel("WER %, greedy decoding")
    _axes[1].set_title("Single-language validation subsets (30 utterances each)")
    _axes[1].legend(fontsize=8)
    _axes[2].plot([r["step"] for r in train_log], [r["loss"] for r in train_log], color="#bbbbbb", label="training loss (running)")
    _axes[2].plot(_steps, [r["val_loss"] for r in evals], "o-", color="#de693c", label="validation loss, mixed")
    _axes[2].plot(_steps, [r["train_loss"] for r in evals], "o--", color="#777777", label="training-subset loss, mixed")
    _axes[2].set_xlabel("optimizer step")
    _axes[2].set_ylabel("cross-entropy per token")
    _axes[2].set_title("Loss")
    _axes[2].legend(fontsize=8)
    _f.tight_layout()
    first, last = evals[0], evals[-1]
    _best = min(evals, key=lambda r: r["val_loss"])
    _rel = lambda key: 100 * (last[key]["wer"] / first[key]["wer"] - 1)  # noqa: E731
    log.fig(10, "learning_curves", _f,
            f"Greedy decoding on fixed subsets. Mixed recordings: validation WER {100 * first['val_mixed']['wer']:.1f}% (step 0 = the untouched "
            f"starting model) → {100 * last['val_mixed']['wer']:.1f}% at step {last['step']}; the training subset "
            f"{100 * first['train_mixed']['wer']:.1f}% → {100 * last['train_mixed']['wer']:.1f}%. Relative change: training {_rel('train_mixed'):+.0f}% "
            f"vs validation {_rel('val_mixed'):+.0f}%; validation loss {first['val_loss']:.3f} → {last['val_loss']:.3f} (lowest {_best['val_loss']:.3f} at "
            f"step {_best['step']}; training-subset loss {first['train_loss']:.3f} → {last['train_loss']:.3f}). "
            "Single-language subsets: " + ", ".join(f"{k.replace('val_', '')} {100 * first[k]['wer']:.1f}% → {100 * last[k]['wer']:.1f}%"
                                                    for k in sorted(first) if k.startswith("val_") and k not in ("val_mixed", "val_loss"))
            + ". A training curve that keeps falling while the validation curve turns up would mean overfitting.")
    {"steps": _steps, "val_mixed": [round(100 * r["val_mixed"]["wer"], 2) for r in evals], "val_loss": [round(r["val_loss"], 3) for r in evals]}
    return


@app.cell
def _(COLORS, LANGUAGES, LANGUAGE_NAMES, RESULTS, SETS, flat, json, log, mixed, np, plt, singles, validation):
    # Step 12 - the chosen checkpoint through the app's decoding, against the baseline and the reference points.
    selection = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
    system = f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"
    ours_single, ours_mixed = validation.single_scores(system), validation.mixed_scores(system)
    _single = {"Kazakh turbo\n(starting model)": (singles["Kazakh turbo\n(starting model)"], "#2a9d8f"),
               "Our large model\n(this round)": (ours_single, "#de693c"),
               "large-v3-turbo\n(untouched)": (singles["large-v3-turbo\n(untouched)"], "#3b6ea8"),
               "Our small model\n(round 1)": (singles["Our small model\n(round 1)"], "#e9b384")}
    _f, _axes = plt.subplots(1, 3, figsize=(18, 4.4), gridspec_kw={"width_ratios": [2.2, 1.6, 1.6]})
    _x = np.arange(len(SETS))
    for _i, (_name, (_scores, _color)) in enumerate(_single.items()):
        _axes[0].bar(_x + (_i - 1.5) * 0.2, [100 * _scores[s]["forced"]["wer"] if s in _scores else np.nan for s in SETS], 0.2, label=flat(_name), color=_color)
    _axes[0].set_xticks(_x, SETS)
    _axes[0].set_ylabel("WER %, language given")
    _axes[0].set_title("Single-language utterances (validation, app decoding)")
    _axes[0].legend(fontsize=8)
    one_pass = {"Kazakh turbo\n(starting)": mixed["Kazakh turbo\n(starting model)"], "large-v3-turbo": mixed["large-v3-turbo\n(untouched)"],
                "Whisper small": mixed["Whisper small\n(untouched)"], "Our small\n(round 1)": mixed["Our small model\n(round 1)"],
                "Our large\n(this round)": ours_mixed}
    pipelines = {"Pipeline,\nround-1 routes": validation.mixed_scores("pipeline:auto"), "Pipeline,\nnew routes": validation.mixed_scores("pipeline-auto-r2"),
                 "Pipeline,\nmixed setting": validation.mixed_scores("pipeline-mixed-r2")}
    _bars = {**one_pass, **{name: block for name, block in pipelines.items() if block is not None}}
    _axes[1].bar(list(_bars), [100 * block["wer"] for block in _bars.values()],
                 color=["#de693c" if "this round" in name else "#555555" if name.startswith("Pipeline") else "#999999" for name in _bars])
    _axes[1].set_ylabel("WER %, automatic language")
    _axes[1].set_title("Mixed recordings (validation): one pass, and the app's pipeline")
    _axes[1].tick_params(axis="x", labelsize=6)
    for _i, _lang in enumerate(LANGUAGES):
        _axes[2].bar(np.arange(len(_bars)) + (_i - 1) * 0.27, [min(130, 100 * block["by_language"][_lang]["wer"]) for block in _bars.values()], 0.27,
                     color=COLORS[_lang], label=LANGUAGE_NAMES[_lang])
    _axes[2].set_xticks(np.arange(len(_bars)), list(_bars), fontsize=6)
    _axes[2].set_ylabel("WER % of the words of each language (cut at 130)")
    _axes[2].set_title("Inside mixed recordings: errors by language")
    _axes[2].legend(fontsize=8)
    _f.tight_layout()
    _rule2 = selection["rule_2_mixed_recognizer"]
    _gap, _from_start = _rule2["large_minus_small_ours"], _rule2["large_minus_starting_model"]
    _report = selection["report_against_starting_model"]
    log.fig(12, "validation_comparison", _f,
            f"Through the app's exact decoding, one-pass validation WER on mixed recordings: starting model {100 * one_pass['Kazakh turbo' + chr(10) + '(starting)']['wer']:.1f}% → "
            f"our large model {100 * ours_mixed['wer']:.1f}% (step {selection['chosen_step']}; difference {100 * _from_start['difference']:+.1f} points, 95% bootstrap "
            f"{100 * _from_start['low']:+.1f} to {100 * _from_start['high']:+.1f}); against our small model of round 1 ({100 * one_pass['Our small' + chr(10) + '(round 1)']['wer']:.1f}%): "
            f"{100 * _gap['difference']:+.1f} points, interval {100 * _gap['low']:+.1f} to {100 * _gap['high']:+.1f}, {_gap['units']} recordings. "
            "With the language given, starting model → our large model: "
            + ", ".join(f"{s} {100 * _report[s]['starting_model_wer']:.1f}% → {100 * _report[s]['cleartalk_large_wer']:.1f}%" for s in SETS)
            + ". The app's pipeline on the same recordings: "
            + ", ".join(f"{flat(name)} {100 * block['wer']:.1f}%" for name, block in pipelines.items() if block is not None)
            + f". Rule 1 (routes, lowest pooled validation WER per language): {selection['rule_1_routes']['routes']}. Rule 2 (recognizer for the "
            f"mixed setting, lowest one-pass WER on these recordings): {_rule2['chosen']}.")
    selection
    return


if __name__ == "__main__":
    app.run()
