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
    sys.path.insert(0, str(ROOT / "ml_pipeline_english"))
    import guard
    from cleartalk_asr.evaluate import RESULTS, bootstrap_difference, eval_tracks, score
    from cleartalk_asr.paths import RUNS

    STATE = ROOT / "ml_pipeline_english"
    return RESULTS, RUNS, STATE, bootstrap_difference, eval_tracks, guard, json, mo, np, pd, plt, score


@app.cell
def _(mo):
    mo.md("""
    # Steps 9-12 · Baselines, training, tuning, validation

    Everything here is **validation only**. The test sets appear only in `04_eval.py`, once.
    """)
    return


@app.cell
def _(RESULTS, eval_tracks, json, score):
    def load_result(set_name, label):
        path = RESULTS / f"{set_name}-{label}-app.json"
        if not path.is_file():
            return None
        saved = json.loads(path.read_text())
        frame = eval_tracks(set_name)
        hypotheses = {r["track_id"]: r["hypothesis"] for r in saved["tracks"]}
        frame = frame.assign(hypothesis=[hypotheses[t] for t in frame.track_id])
        metrics, units = score(frame, frame.hypothesis)
        legacy, _ = score(frame, frame.hypothesis, "legacy")
        return {"metrics": metrics, "legacy": legacy, "units": units, "frame": frame, "seconds": saved["seconds"]}

    results = {(s, m): load_result(s, m) for s in ("val", "v4val") for m in ("baseline", "finetuned")}
    {k: (v["metrics"] if v else None) for k, v in results.items()}
    return (results,)


@app.cell
def _(STATE, guard, plt, results):
    _base, _v4 = results[("val", "baseline")], results[("v4val", "baseline")]
    _labels = ["dummy: empty text", "pretrained · V1 mixtures", "pretrained · V2 (v4) mixtures", "pretrained · V3 clean"]
    _values = [100.0, 100 * _base["metrics"]["pairs"]["wer"], 100 * _v4["metrics"]["pairs"]["wer"],
               100 * _base["metrics"]["singles"]["wer"]]
    _f, _ax = plt.subplots(figsize=(8, 3.4))
    _bars = _ax.barh(_labels[::-1], _values[::-1], color=["#5a9e5a", "#d08c2b", "#d08c2b", "#999999"])
    for _bar, _v in zip(_bars, _values[::-1]):
        _ax.text(_v, _bar.get_y() + _bar.get_height() / 2, f" {_v:.1f}%", va="center", fontsize=9)
    _ax.set_xlabel("validation WER (%) · app decoding: faster-whisper int8, beam 5, VAD")
    _ax.set_title("Step 9 · baselines to beat")
    guard.fig(9, "baselines", _f,
              f"An empty transcript scores 100% WER. The current app (pretrained Whisper small.en on separated tracks) has "
              f"{_values[1]:.1f}% WER on {_base['metrics']['pairs']['mixtures']} new validation mixtures, {_values[2]:.1f}% on the 48 "
              f"v4 validation mixtures, and {_values[3]:.1f}% on clean single-speaker speech. The gap between clean and separated "
              "speech is what fine-tuning tries to close.",
              state_dir=STATE)
    return


@app.cell
def _(RUNS, json, pd):
    def history(run):
        path = RUNS / run / "history.jsonl"
        if not path.is_file():
            return None, None
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        # A resumed run can re-log a few steps after its last saved state: keep the latest record.
        rows = list({(r["type"], r["step"]): r for r in rows}.values())
        train = pd.DataFrame([r for r in rows if r["type"] == "train"])
        evals = pd.DataFrame([{"step": r["step"], "epoch": r["epoch"], "val_loss": r.get("val_loss"),
                               "train_eval_loss": r.get("train_eval_loss"),
                               **{f"{k}_wer": v["wer"] for k, v in r.items() if isinstance(v, dict) and "wer" in v}}
                              for r in rows if r["type"] == "eval"])
        return train, evals

    runs = sorted(p.name for p in RUNS.glob("*") if (p / "history.jsonl").is_file() and not p.name.startswith("smoke"))
    histories = {run: history(run) for run in runs}
    runs
    return histories, runs


@app.cell
def _(RUNS, STATE, guard, histories, json, plt):
    _selected = RUNS / "selected.json"
    _main = json.loads(_selected.read_text())["run"] if _selected.is_file() else None
    if _main:
        _train, _evals = histories[_main]
        _best = json.loads((RUNS / _main / "best.json").read_text())
        _f, _axes = plt.subplots(1, 2, figsize=(13, 3.8))
        _smooth = _train.set_index("step")["loss"].rolling(10, min_periods=1).mean()
        _axes[0].plot(_smooth.index, _smooth.values, color="#9db7d6", linewidth=1,
                      label="training batches (with augmentation, moving average)")
        _axes[0].plot(_evals.step, _evals.train_eval_loss, "o--", color="#3b6ea8", label="training subset (evaluation mode)")
        _axes[0].plot(_evals.step, _evals.val_loss, "o-", color="#d08c2b", label="validation mixtures")
        _axes[0].set_xlabel("optimizer step")
        _axes[0].set_ylabel("cross-entropy per token")
        _axes[0].legend(fontsize=8)
        _axes[0].set_title("Loss")
        _axes[1].plot(_evals.step, 100 * _evals.val_pairs_wer, "o-", color="#d08c2b", label="validation mixtures")
        _axes[1].plot(_evals.step, 100 * _evals.train_pairs_wer, "o--", color="#3b6ea8", label="training mixtures (fixed subset)")
        _axes[1].plot(_evals.step, 100 * _evals.val_singles_wer, "s-", color="#5a9e5a", label="validation clean speech")
        _axes[1].axvline(_best["step"], color="black", linestyle=":", linewidth=1)
        _axes[1].text(_best["step"], _axes[1].get_ylim()[1] * 0.95, " selected", fontsize=8)
        _axes[1].set_xlabel("optimizer step")
        _axes[1].set_ylabel("WER (%) · greedy decoding")
        _axes[1].legend(fontsize=8)
        _axes[1].set_title("Word error rate during training")
        _last = _evals.iloc[-1]
        _first = _evals.iloc[0]
        _train_change = 100 * (_last.train_pairs_wer / _first.train_pairs_wer - 1)
        _val_change = 100 * (_last.val_pairs_wer / _first.val_pairs_wer - 1)
        guard.fig(10, "learning_curves", _f,
                  f"Validation WER on mixtures goes from {100 * _first.val_pairs_wer:.1f}% (step 0 = pretrained) to "
                  f"{100 * _best['val_pairs_wer']:.1f}% at step {_best['step']}; clean-speech WER {100 * _first.val_singles_wer:.1f}% → "
                  f"{100 * _last.val_singles_wer:.1f}%. The training subset is easier from the start ({100 * _first.train_pairs_wer:.1f}% at "
                  "step 0, because failed separations are filtered out of training), so the overfitting check is the relative "
                  f"change: training {_train_change:+.0f}% vs validation {_val_change:+.0f}%. The gap does not open and validation "
                  f"loss keeps falling ({_first.val_loss:.3f} → {_last.val_loss:.3f}): no sign of overfitting within one epoch.",
                  state_dir=STATE)
    return


@app.cell
def _(STATE, guard, histories, plt):
    _pilots = {run: h for run, h in histories.items() if run.startswith("lr")}
    if _pilots:
        _f, _ax = plt.subplots(figsize=(7, 3.4))
        _summary = {}
        _pilot_step = 100
        for _run, (_train, _evals) in sorted(_pilots.items()):
            _early = _evals[_evals.step <= _pilot_step]
            _ax.plot(_early.step, 100 * _early.val_pairs_wer, "o-", label=_run)
            _summary[_run] = 100 * _early.val_pairs_wer.iloc[-1]
        _ax.set_xlabel("optimizer step")
        _ax.set_ylabel("validation WER (%) · greedy")
        _ax.legend()
        _ax.set_title("Step 11 · learning-rate pilots (validation only)")
        _winner = min(_summary, key=_summary.get)
        guard.fig(11, "learning_rate_pilots", _f,
                  f"Identical {_pilot_step}-step runs that differ only in learning rate, compared on validation mixtures: "
                  + ", ".join(f"{k} → {v:.1f}%" for k, v in _summary.items())
                  + f". {_winner} was continued as the full run; the test sets played no part.",
                  state_dir=STATE)
    return


@app.cell
def _(STATE, bootstrap_difference, guard, np, plt, results):
    _b, _t = results[("val", "baseline")], results[("val", "finetuned")]
    if _t is not None:
        _vb, _vt = results[("v4val", "baseline")], results[("v4val", "finetuned")]
        _pairs = lambda r: r["units"][r["units"].kind == "pair"]  # noqa: E731
        _ci = bootstrap_difference(_pairs(_b), _pairs(_t))
        _f, _axes = plt.subplots(1, 2, figsize=(13, 4))
        _names = ["V1 mixtures", "V2 (v4) mixtures", "V3 clean speech"]
        _base = [_b["metrics"]["pairs"]["wer"], _vb["metrics"]["pairs"]["wer"], _b["metrics"]["singles"]["wer"]]
        _tuned = [_t["metrics"]["pairs"]["wer"], _vt["metrics"]["pairs"]["wer"], _t["metrics"]["singles"]["wer"]]
        _x = np.arange(3)
        _axes[0].bar(_x - 0.2, 100 * np.array(_base), 0.4, color="#999999", label="pretrained")
        _axes[0].bar(_x + 0.2, 100 * np.array(_tuned), 0.4, color="#3b6ea8", label="ClearTalk fine-tuned")
        for _i in range(3):
            _axes[0].text(_x[_i] - 0.2, 100 * _base[_i], f"{100 * _base[_i]:.1f}", ha="center", va="bottom", fontsize=8)
            _axes[0].text(_x[_i] + 0.2, 100 * _tuned[_i], f"{100 * _tuned[_i]:.1f}", ha="center", va="bottom", fontsize=8)
        _axes[0].set_xticks(_x, _names)
        _axes[0].set_ylabel("WER (%) · app decoding")
        _axes[0].legend()
        _axes[0].set_title("Validation, same audio for both models")
        _fb, _ft = _b["frame"], _t["frame"]
        _units_b = _b["units"].set_index("recipe_id")
        _units_t = _t["units"].set_index("recipe_id")
        _cond = _fb[_fb.kind == "pair"].drop_duplicates("recipe_id").set_index("recipe_id")[["snr_db", "noise"]]
        _cond["noise"] = _cond.noise.where(_cond.noise == "esc10", "synthetic")
        _rows = []
        for _col in ("snr_db", "noise"):
            for _value, _ids in _cond.groupby(_col).groups.items():
                _eb, _et = _units_b.loc[_ids], _units_t.loc[_ids]
                _rows.append((f"{_col.replace('_db', '')}={_value:g}" if _col == "snr_db" else str(_value),
                               _eb.errors.sum() / _eb.reference_words.sum(), _et.errors.sum() / _et.reference_words.sum()))
        _y = np.arange(len(_rows))
        _axes[1].barh(_y + 0.2, [100 * r[1] for r in _rows], 0.4, color="#999999", label="pretrained")
        _axes[1].barh(_y - 0.2, [100 * r[2] for r in _rows], 0.4, color="#3b6ea8", label="ClearTalk")
        _axes[1].set_yticks(_y, [r[0] for r in _rows])
        _axes[1].set_xlabel("WER (%) on V1 mixtures")
        _axes[1].set_title("By noise level and noise type")
        _axes[1].legend(fontsize=8)
        guard.fig(12, "validation_comparison", _f,
                  f"Through the app's exact decoding, validation WER on mixtures is {100 * _base[0]:.1f}% → {100 * _tuned[0]:.1f}% "
                  f"(difference {100 * _ci['difference']:+.1f} points, 95% bootstrap interval {100 * _ci['low']:+.1f} to "
                  f"{100 * _ci['high']:+.1f} over {_ci['mixtures']} mixtures); v4 validation {100 * _base[1]:.1f}% → {100 * _tuned[1]:.1f}%; "
                  f"clean speech {100 * _base[2]:.1f}% → {100 * _tuned[2]:.1f}%. The right panel shows where the gain comes from.",
                  state_dir=STATE)
    return


if __name__ == "__main__":
    app.run()
