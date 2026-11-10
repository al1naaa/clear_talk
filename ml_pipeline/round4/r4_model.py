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
    from cleartalk_ml import round4
    from cleartalk_ml.paths import RESULTS, RUNS

    log.use(round4.R4)
    selection = json.loads((RESULTS / "selection-round4.json").read_text(encoding="utf-8"))
    return RESULTS, RUNS, json, log, mo, np, plt, selection


@app.cell
def _(mo):
    mo.md("""
    # Round 4 · Steps 9-12: training in noise and the decision on validation

    Everything here is measured on validation data. Test D stays frozen and unopened.
    """)
    return


@app.cell
def _(RUNS, json, log, plt, selection):
    # Step 10 - learning curves: clean and noisy monitors side by side (step 0 = the round-2 model).
    history = [json.loads(line) for line in (RUNS / selection["run"] / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {}
    for _r in history:
        if _r.get("type") == "eval":
            seen[_r["step"]] = _r
    evals = [seen[k] for k in sorted(seen)]
    _steps = [r["step"] for r in evals]
    train_log = [r for r in history if r.get("type") == "train"]
    _f, _axes = plt.subplots(1, 2, figsize=(14, 4))
    _axes[0].plot(_steps, [100 * r["val_mixed"]["wer"] for r in evals], "o-", color="#2a9d8f", label="validation mixed, clean")
    _axes[0].plot(_steps, [100 * r["valnoise_mixed"]["wer"] for r in evals], "o-", color="#de693c", label="validation mixed, babble 10 dB")
    for _key, _label in (("val_fleurs-kk", "FLEURS kk"), ("val_fleurs-ru", "FLEURS ru"), ("val_fleurs-en", "FLEURS en")):
        _axes[0].plot(_steps, [100 * r[_key]["wer"] for r in evals], ".--", label=_label)
    _axes[0].set_xlabel("optimizer step (0 = the round-2 model)")
    _axes[0].set_ylabel("WER %, greedy decoding")
    _axes[0].set_title("Monitors: clean and noisy validation subsets")
    _axes[0].legend(fontsize=8)
    _axes[1].plot([r["step"] for r in train_log], [r["loss"] for r in train_log], color="#bbbbbb", label="training loss (half of it noisy)")
    _axes[1].plot(_steps, [r["val_loss"] for r in evals], "o-", color="#2a9d8f", label="validation loss, clean mixed")
    _axes[1].plot(_steps, [r["valnoise_loss"] for r in evals], "o-", color="#de693c", label="validation loss, babble 10 dB")
    _axes[1].set_xlabel("optimizer step")
    _axes[1].set_ylabel("cross-entropy per token")
    _axes[1].set_title("Loss")
    _axes[1].legend(fontsize=8)
    _f.tight_layout()
    first, last = evals[0], evals[-1]
    log.fig(10, "learning_curves_noise", _f,
            f"Greedy decoding on fixed validation subsets; step 0 is the round-2 model. Mixed recordings with babble at 10 dB: "
            f"{100 * first['valnoise_mixed']['wer']:.1f}% -> {100 * last['valnoise_mixed']['wer']:.1f}% at step {last['step']}; the same recordings clean: "
            f"{100 * first['val_mixed']['wer']:.1f}% -> {100 * last['val_mixed']['wer']:.1f}%. FLEURS kk/ru/en: "
            + ", ".join(f"{100 * first[k]['wer']:.1f}% -> {100 * last[k]['wer']:.1f}%" for k in ("val_fleurs-kk", "val_fleurs-ru", "val_fleurs-en"))
            + f". Validation loss clean {first['val_loss']:.3f} -> {last['val_loss']:.3f}, babble {first['valnoise_loss']:.3f} -> {last['valnoise_loss']:.3f}.")
    {"steps": _steps, "noisy": [round(100 * r["valnoise_mixed"]["wer"], 2) for r in evals], "clean": [round(100 * r["val_mixed"]["wer"], 2) for r in evals]}
    return


@app.cell
def _(log, np, plt, selection):
    # Step 12 - the decision: the round-2 model and the evaluated checkpoints on the noisy robustness recordings,
    # and the chosen checkpoint against the round-2 model on the clean guards.
    _f, _axes = plt.subplots(1, 2, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1.4]})
    _names = ["round 2"] + [f"step {s}" for s in selection["checkpoints_noisy_wer"]]
    _values = [100 * selection["baseline_noisy_wer"]] + [100 * v for v in selection["checkpoints_noisy_wer"].values()]
    _axes[0].bar(_names, _values, color=["#999999"] + ["#de693c"] * (len(_names) - 1))
    _axes[0].set_ylabel("pooled WER %, one pass (noisy validation)")
    _axes[0].set_title("Noisy robustness recordings: white 20/10 dB, babble 10/5 dB")
    chosen = selection.get("chosen_step")
    if chosen is not None:
        _guard = selection["clean_guard"][str(chosen)]
        _keys = list(_guard)
        _x = np.arange(len(_keys))
        _axes[1].bar(_x - 0.2, [100 * _guard[k]["round2"] for k in _keys], 0.4, color="#999999", label="round 2")
        _axes[1].bar(_x + 0.2, [100 * _guard[k]["candidate"] for k in _keys], 0.4, color="#de693c", label=f"round 4, step {chosen}")
        _axes[1].set_xticks(_x, _keys)
        _axes[1].set_ylabel("WER %, clean validation")
        _axes[1].set_title("Clean guard (no set more than 0.5 points worse)")
        _axes[1].legend(fontsize=8)
    _f.tight_layout()
    caption = ("Pooled one-pass WER over the noisy robustness validation recordings: round-2 model "
               f"{100 * selection['baseline_noisy_wer']:.1f}%; checkpoints " + ", ".join(f"step {s} {100 * v:.1f}%" for s, v in selection["checkpoints_noisy_wer"].items()) + ". ")
    if chosen is not None:
        caption += (f"Chosen: step {chosen} ({100 * selection['noisy']['relative_change']:+.0f}% relative, difference {100 * selection['noisy']['difference']:+.1f} "
                    f"points, 95% {100 * selection['noisy']['low']:+.1f} to {100 * selection['noisy']['high']:+.1f}); by condition "
                    + ", ".join(f"{c} {100 * v['round2']:.1f}% -> {100 * v['candidate']:.1f}%" for c, v in selection["by_condition"].items())
                    + ". Clean guard: " + ", ".join(f"{k} {100 * v['round2']:.1f}% -> {100 * v['candidate']:.1f}%" for k, v in selection["clean_guard"][str(chosen)].items())
                    + f". Promotion: {'yes' if selection['promoted'] else 'no'}.")
    else:
        caption += "No checkpoint met the clean guard; the round-2 model stays."
    log.fig(12, "noise_decision", _f, caption)
    selection
    return


if __name__ == "__main__":
    app.run()
