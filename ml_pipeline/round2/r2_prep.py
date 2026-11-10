import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(ROOT))
    os.environ["CLEARTALK_BASE"] = "kazakh-turbo"  # this round's base model: 128 mel bands, the large tokenizer
    from cleartalk_ml import pipeline_log as log
    from cleartalk_ml import round2, splits, validation
    from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES

    log.use(round2.R2)
    COLORS = {"kk": "#2a9d8f", "ru": "#3b6ea8", "en": "#d08c2b"}
    return COLORS, Counter, LANGUAGES, LANGUAGE_NAMES, log, mo, np, plt, round2, splits, validation


@app.cell
def _(mo):
    mo.md("""
    # Round 2 · Steps 1-8: what is new before training the large model

    The speech data, the cleaning rules, the training examples and the validation sets are those of round 1
    (`../01_eda.py`, `../02_prep.py`). New here: where the candidate starts from, a second frozen test set,
    and the large model's own features and tokens. No model is fitted in this notebook.
    """)
    return


@app.cell
def _(log, np, plt, validation):
    # Steps 1-2 - the starting points, on VALIDATION (transcripts cached in round 1; nothing is transcribed here).
    systems = {"Whisper small\n(untouched)": "small", "Our small model\n(round 1)": "runs/lr5e-4/ct2-step-00900",
               "large-v3-turbo\n(untouched)": "turbo", "Kazakh turbo\n(starting model)": "kazakh"}
    single = {name: validation.single_scores(system) for name, system in systems.items()}
    mixed = {name: validation.mixed_scores(system) for name, system in systems.items()}
    _sets = ["fleurs-kk", "ksc2-kk", "fleurs-ru", "golos-ru", "fleurs-en"]
    _f, _axes = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [2.4, 1]})
    _x = np.arange(len(_sets))
    for _i, (_name, _color) in enumerate(zip(systems, ("#999999", "#de693c", "#3b6ea8", "#2a9d8f"))):
        _vals = [min(110, 100 * single[_name][s]["forced"]["wer"]) if s in single[_name] else np.nan for s in _sets]
        _axes[0].bar(_x + (_i - 1.5) * 0.2, _vals, 0.2, color=_color, label=_name.replace("\n", " "))
    _axes[0].set_xticks(_x, _sets)
    _axes[0].set_ylabel("WER %, language given (cut at 110)")
    _axes[0].set_title("Single-language validation sets")
    _axes[0].legend(fontsize=8)
    _names = [name for name in systems if mixed[name] is not None]
    _axes[1].bar([n for n in _names], [100 * mixed[n]["wer"] for n in _names],
                 color=[c for n, c in zip(systems, ("#999999", "#de693c", "#3b6ea8", "#2a9d8f")) if mixed[n] is not None])
    _axes[1].set_ylabel("WER %, one pass, automatic language")
    _axes[1].set_title("Mixed-language validation recordings")
    _axes[1].tick_params(axis="x", labelsize=7)
    _f.tight_layout()
    _k = single["Kazakh turbo\n(starting model)"]
    _t = single["large-v3-turbo\n(untouched)"]
    log.fig(2, "starting_points_validation", _f,
            "Where this round starts, on validation. The Kazakh turbo fine-tune is far ahead on Kazakh "
            f"(FLEURS {100 * _k['fleurs-kk']['forced']['wer']:.1f}%, KSC2 {100 * _k['ksc2-kk']['forced']['wer']:.1f}%) but behind the untouched "
            f"large-v3-turbo on Russian ({100 * _k['fleurs-ru']['forced']['wer']:.1f}% vs {100 * _t['fleurs-ru']['forced']['wer']:.1f}%) and English "
            f"({100 * _k['fleurs-en']['forced']['wer']:.1f}% vs {100 * _t['fleurs-en']['forced']['wer']:.1f}%); it was never scored on Golos or on mixed "
            "recordings (it is routed Kazakh only). On mixed recordings in one pass: "
            + ", ".join(f"{n.replace(chr(10), ' ')} {100 * mixed[n]['wer']:.1f}%" for n in _names)
            + ". The large models have the acoustic quality; what they lack is the habit of following a language change, "
            "which the decoder decides - the reason to train the decoder only.")
    {name.replace("\n", " "): {s: round(100 * block["forced"]["wer"], 1) for s, block in scores.items()} for name, scores in single.items()}
    return


@app.cell
def _(Counter, log, np, plt, round2, splits):
    # Steps 4-6 - cleaning and roles are round 1's; the new part is test B.
    round2.build_mixed_plan()
    units_b = round2.test_b_units()
    fresh = round2.check_fresh(units_b)
    frozen = round2.freeze(units_b)
    units_a, units_val = splits.eval_units("test"), splits.eval_units("val")
    _sets = sorted(set(units_b.set))
    _f, _axes = plt.subplots(1, 2, figsize=(13, 3.8))
    _x = np.arange(len(_sets))
    for _i, (_label, _frame, _color) in enumerate((("validation", units_val, "#d08c2b"), ("test A (spent)", units_a, "#bbbbbb"),
                                                   ("test B (frozen)", units_b, "#5a9e5a"))):
        _counts = Counter(_frame.set)
        _axes[0].bar(_x + (_i - 1) * 0.27, [_counts[s] for s in _sets], 0.27, color=_color, label=_label)
    _axes[0].set_xticks(_x, _sets)
    _axes[0].set_ylabel("units")
    _axes[0].set_title("Evaluation units per set")
    _axes[0].legend()
    _mixed = units_b[units_b.kind == "mixed"]
    _parts = Counter(len(languages.split("+")) for languages in _mixed.languages)
    _axes[1].bar([str(k) for k in sorted(_parts)], [_parts[k] for k in sorted(_parts)], color="#5a9e5a")
    _axes[1].set_xlabel("utterances in one mixed recording of test B")
    _axes[1].set_ylabel("recordings")
    _axes[1].set_title(f"Mixed recordings of test B (median {np.median(_mixed.duration):.1f} s)")
    _f.tight_layout()
    log.fig(6, "test_b", _f,
            f"Test B: {frozen['n_test']:,} units, {round2.PER_SET} per set, frozen by fingerprint before any training of this round. Every unit "
            "comes from test-role utterances that the first final test never transcribed; shared with test A, validation or "
            f"training: {sum(fresh.values())}. It is smaller than test A (300 per set) because only 347 unused FLEURS English test "
            "utterances are left. Cleaning rules, training examples and validation sets are unchanged from round 1.")
    (fresh, frozen)
    return


@app.cell
def _(LANGUAGE_NAMES, log, np, plt, splits):
    # Step 7 - features of the large model: Whisper's fixed log-mel spectrogram with 128 bands.
    from cleartalk_ml.dataset import example_audio, features

    train_examples = splits.training_examples()
    _example = next(e for e in train_examples if e["kind"] == "mixed" and len(e["parts"]) == 3)
    _mel = features(example_audio(_example))
    _f, _ax = plt.subplots(figsize=(13, 3))
    _ax.imshow(_mel, origin="lower", aspect="auto", extent=(0, 30, 0, _mel.shape[0]), cmap="magma")
    for _part in _example["parts"]:
        _ax.axvline(_part["start"], color="white", linewidth=1)
        _ax.text(_part["start"] + 0.2, _mel.shape[0] * 0.87, LANGUAGE_NAMES[_part["language"]], color="white", fontsize=9)
    _ax.set_xlabel("seconds")
    _ax.set_ylabel("mel band")
    _ax.set_title("The same mixed training recording as the large model sees it")
    log.fig(7, "logmel_large_model", _f,
            f"The large model's input is Whisper's fixed log-mel spectrogram with {_mel.shape[0]} bands × {_mel.shape[1]} frames (the small "
            "model uses 80 bands). It is a fixed formula with no statistics estimated from our data, so nothing can leak from "
            "validation or test. The training examples themselves are the 14,400 of round 1.")
    return (train_examples,)


@app.cell
def _(log, np, plt, train_examples):
    # Step 8 - targets with the large model's tokenizer.
    from cleartalk_ml.dataset import MAX_LABEL_TOKENS, target_tokens, uses_timestamps

    _lengths = {"single": [], "mixed": []}
    for _example in train_examples:
        _lengths[_example["kind"]].append(len(target_tokens(_example, uses_timestamps(_example["id"]))))
    _f, _ax = plt.subplots(figsize=(7.5, 3.4))
    for _kind, _color in (("single", "#3b6ea8"), ("mixed", "#de693c")):
        _ax.hist(_lengths[_kind], bins=np.linspace(0, MAX_LABEL_TOKENS, 56), histtype="step", linewidth=1.6, color=_color,
                 label=f"{_kind} ({len(_lengths[_kind]):,})")
    _ax.axvline(MAX_LABEL_TOKENS, color="black", linewidth=0.8)
    _ax.set_xlabel("target length in tokens (black line = limit)")
    _ax.set_title("Training targets with the large model's tokenizer")
    _ax.legend()
    _all = np.array(_lengths["single"] + _lengths["mixed"])
    log.fig(8, "target_tokens_large_model", _f,
            f"Targets with the large model's tokenizer (the same vocabulary as large-v3-turbo; token ids of the special tokens "
            f"differ from the small model's): median {int(np.median(_lengths['single']))} tokens for a single utterance, "
            f"{int(np.median(_lengths['mixed']))} for a mixed recording, longest {int(_all.max())}; {int((_all >= MAX_LABEL_TOKENS).sum())} reach the limit. "
            "The target format is round 1's: one language token per window, half of the examples with timestamps, language and "
            "task tokens left out of the loss.")
    return


if __name__ == "__main__":
    app.run()
