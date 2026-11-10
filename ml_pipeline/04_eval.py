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
    from cleartalk_ml import analysis
    from cleartalk_ml import pipeline_log as log
    from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES, RESULTS

    selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    OURS = f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"
    COLORS = {"kk": "#2a9d8f", "ru": "#3b6ea8", "en": "#d08c2b"}
    return COLORS, LANGUAGES, LANGUAGE_NAMES, OURS, RESULTS, analysis, json, log, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Steps 13-16 · Error analysis, the one-time final test, deployment and monitoring

    Step 13 looks only at **validation** transcripts. The test set is opened once, by
    `python -m cleartalk_ml.final_test` (it needs Gate C); this notebook only draws its saved result.
    """)
    return


@app.cell
def _(LANGUAGES, LANGUAGE_NAMES, OURS, analysis, log, np, plt):
    # Step 13 - where the errors are (validation, the app's decoding path).
    moves = analysis.better_same_worse("mixed-val", "small", OURS)
    kinds = {name: analysis.edit_kinds("mixed-val", system) for name, system in (("Whisper small", "small"), ("ours", OURS))}
    position = {name: analysis.by_position(system) for name, system in (("Whisper small", "small"), ("ours", OURS))}
    script = {name: analysis.wrong_script(system) for name, system in (("Whisper small", "small"), ("ours", OURS))}
    sources = {name: analysis.ksc2_by_source(system) for name, system in (("Whisper small", "small"), ("ours", OURS))}
    prompt = analysis.prompt_effect(OURS)
    _f, _axes = plt.subplots(2, 2, figsize=(13, 8.5))
    _axes = _axes.ravel()
    _x = np.arange(3)
    for _i, (_name, _color) in enumerate((("Whisper small", "#999999"), ("ours", "#de693c"))):
        _axes[0].bar(_x + (_i - 0.5) * 0.4, [kinds[_name][k] for k in ("substitutions", "deletions", "insertions")], 0.4, color=_color, label=_name)
    _axes[0].set_xticks(_x, ["wrong word", "missing word", "extra word"])
    _axes[0].set_ylabel("words")
    _axes[0].set_title(f"Mixed recordings: kinds of errors\n({kinds['ours']['reference_words']:,} reference words)", fontsize=10)
    _axes[0].legend()
    _keys = list(position["ours"])
    for _i, (_name, _color) in enumerate((("Whisper small", "#999999"), ("ours", "#de693c"))):
        _axes[1].bar(np.arange(len(_keys)) + (_i - 0.5) * 0.4, [100 * position[_name][k] for k in _keys], 0.4, color=_color)
    _axes[1].set_xticks(np.arange(len(_keys)), [k.replace("part", "utterance") for k in _keys])
    _axes[1].set_ylabel("WER %")
    _axes[1].set_title("Mixed recordings: errors by place in the recording", fontsize=10)
    for _i, (_name, _color) in enumerate((("Whisper small", "#999999"), ("ours", "#de693c"))):
        _axes[2].bar(np.arange(3) + (_i - 0.5) * 0.4, [100 * script[_name][lang] for lang in LANGUAGES], 0.4, color=_color)
    _axes[2].set_xticks(np.arange(3), [LANGUAGE_NAMES[lang] for lang in LANGUAGES])
    _axes[2].set_ylabel("% of the language's words")
    _axes[2].set_title("Mixed recordings: words written in the other alphabet", fontsize=10)
    _names = sorted(sources["ours"])
    for _i, (_name, _color) in enumerate((("Whisper small", "#999999"), ("ours", "#de693c"))):
        _axes[3].bar(np.arange(len(_names)) + (_i - 0.5) * 0.4, [min(150, 100 * sources[_name][s]["wer"]) for s in _names], 0.4, color=_color)
    _axes[3].set_xticks(np.arange(len(_names)), [f"{s}\n({sources['ours'][s]['units']})" for s in _names], fontsize=8)
    _axes[3].set_ylabel("WER %, language given (cut at 150)")
    _axes[3].set_title("Kazakh KSC2 by source\n(training saw only 'crowdsourced')", fontsize=10)
    _f.tight_layout()
    log.fig(13, "error_analysis_validation", _f,
            f"On {moves['units']} mixed validation recordings our model makes fewer errors than Whisper small on {moves['better']}, the same on "
            f"{moves['same']} and more on {moves['worse']}. Wrong words {kinds['Whisper small']['substitutions']:,} → {kinds['ours']['substitutions']:,}, "
            f"missing {kinds['Whisper small']['deletions']:,} → {kinds['ours']['deletions']:,}, extra {kinds['Whisper small']['insertions']:,} → "
            f"{kinds['ours']['insertions']:,}. By place in the recording: " + ", ".join(f"{k.replace('part', 'utterance')} {100 * position['Whisper small'][k]:.0f}% → {100 * position['ours'][k]:.0f}%" for k in _keys)
            + ". Words written in the other alphabet (the untouched model keeps writing in the first language it detected): "
            + ", ".join(f"{LANGUAGE_NAMES[lang]} {100 * script['Whisper small'][lang]:.1f}% → {100 * script['ours'][lang]:.1f}%" for lang in LANGUAGES)
            + ". Kazakh KSC2 by source: " + ", ".join(f"{s} {100 * sources['Whisper small'][s]['wer']:.0f}% → {100 * sources['ours'][s]['wer']:.0f}%" for s in _names)
            + f". The language detected for a mixed recording is its first utterance's in {100 * prompt['first_language_detected']:.0f}% of cases"
            + (f" (WER {100 * prompt['wer_when_first']:.1f}% then, {100 * prompt['wer_when_other']:.1f}% otherwise)" if prompt["wer_when_other"] is not None else "")
            + ".")
    {"moves": moves, "kinds": kinds, "position": position, "wrong_script": script, "prompt": prompt}
    return


@app.cell
def _(OURS, analysis):
    # The recordings where our model is worst (validation): read them before trusting an average.
    analysis.worst("mixed-val", OURS, 8)
    return


@app.cell
def _(LANGUAGES, LANGUAGE_NAMES, RESULTS, json, log, np, pd, plt):
    # Step 14 - the final test, drawn from the saved result of the one sanctioned run.
    _path = RESULTS / "test-final.json"
    final = json.loads(_path.read_text(encoding="utf-8")) if _path.is_file() else None
    if final:
        _scores = pd.DataFrame(final["scores"])
        _sets = ["mixed", "fleurs-kk", "fleurs-ru", "fleurs-en", "ksc2-kk", "golos-ru"]
        _systems = [("small", "Whisper small (untouched)", "#999999"), ("cleartalk", "ClearTalk (ours)", "#de693c"),
                    ("turbo", "Whisper large-v3-turbo", "#3b6ea8"), ("kazakh", "Kazakh turbo fine-tune", "#2a9d8f"),
                    ("pipeline:auto", "App pipeline, ready models", "#7b5ea7")]
        _f, _axes = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})

        def _value(system, set_name):
            row = _scores[(_scores.system == system) & (_scores.set == set_name)]
            if not len(row):
                return np.nan
            key = "wer_auto" if set_name == "mixed" else "wer_forced"
            return 100 * float(row.iloc[0][key])

        _x = np.arange(len(_sets))
        for _i, (_key, _label, _color) in enumerate(_systems):
            # A system that was not measured on a set has no bar (NaN must not be clipped into a full bar).
            _heights = [np.nan if np.isnan(_value(_key, s)) else min(130, _value(_key, s)) for s in _sets]
            _axes[0].bar(_x + (_i - 2) * 0.16, _heights, 0.16, color=_color, label=_label)
        _axes[0].set_xticks(_x, _sets)
        _axes[0].set_ylabel("WER % (mixed: automatic language; others: language given; cut at 130)")
        _axes[0].set_title("Final test, one run: every system on the same frozen audio")
        _axes[0].legend(fontsize=8)
        _by = final["mixed_by_language"]
        _names = [k for k, _, _ in _systems if k in _by]
        for _i, _lang in enumerate(LANGUAGES):
            _axes[1].bar(np.arange(len(_names)) + (_i - 1) * 0.27, [min(130, 100 * _by[n][_lang]["wer"]) for n in _names], 0.27,
                         label=LANGUAGE_NAMES[_lang], color={"kk": "#2a9d8f", "ru": "#3b6ea8", "en": "#d08c2b"}[_lang])
        _axes[1].set_xticks(np.arange(len(_names)), [dict((k, l) for k, l, _ in _systems)[n].split(" (")[0].replace(", ready models", "") for n in _names], fontsize=7)
        _axes[1].set_title("Mixed test recordings: errors by language", fontsize=10)
        _axes[1].set_ylabel("WER % of the words of each language (cut at 130)")
        _axes[1].legend(fontsize=8)
        _f.tight_layout()
        _cmp = final["cleartalk_vs_small"]
        log.fig(14, "final_test", _f,
                "Final test, opened once, same frozen audio for every system. Our model against untouched Whisper small: "
                + "; ".join(f"{s} {_value('small', s):.1f}% → {_value('cleartalk', s):.1f}% (difference {100 * _cmp[s]['difference']:+.1f} points, "
                            f"95% bootstrap {100 * _cmp[s]['low']:+.1f} to {100 * _cmp[s]['high']:+.1f}, {_cmp[s]['units']} units)" for s in _sets)
                + ". Reference points: " + "; ".join(f"{label} " + ", ".join(f"{s} {_value(key, s):.1f}%" for s in _sets if not np.isnan(_value(key, s)))
                                                     for key, label, _ in _systems[2:])
                + ". Reported as measured; the model and the app's routes were fixed on validation before this run.")
    final["cleartalk_vs_small"] if final else "The final test has not been run yet."
    return


@app.cell
def _(RESULTS, json):
    # Step 15 - deployment check written by scripts/check_multilingual_deployment.py.
    _path = RESULTS / "deployment-check.json"
    json.loads(_path.read_text(encoding="utf-8")) if _path.is_file() else "The deployment check has not been run yet."
    return


if __name__ == "__main__":
    app.run()
