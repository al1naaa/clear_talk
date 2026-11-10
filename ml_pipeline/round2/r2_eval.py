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

    ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(ROOT))
    from cleartalk_ml import analysis
    from cleartalk_ml import pipeline_log as log
    from cleartalk_ml import round2
    from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES, RESULTS

    log.use(round2.R2)
    selection = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
    round1 = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    OURS = f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"
    # name on the figures -> (system in the validation transcript files, colour)
    THREE = {"starting model": ("kazakh", "#2a9d8f"), "our small (round 1)": (f"runs/{round1['run']}/ct2-step-{round1['chosen_step']:05d}", "#e9b384"),
             "our large": (OURS, "#de693c")}
    return LANGUAGES, LANGUAGE_NAMES, OURS, RESULTS, THREE, analysis, json, log, mo, np, pd, plt, selection


@app.cell
def _(mo):
    mo.md("""
    # Round 2 · Steps 13-16: error analysis, the one-time final test on test B, deployment and monitoring

    Step 13 looks only at **validation** transcripts. Test B is opened once, by
    `python -m cleartalk_ml.final_test_b` (it needs Gate C); this notebook only draws its saved result.
    """)
    return


@app.cell
def _(LANGUAGES, LANGUAGE_NAMES, OURS, THREE, analysis, log, np, plt, selection):
    # Step 13 - where the errors are (validation, the app's decoding path).
    moves = {name: analysis.better_same_worse("mixed-val", system, OURS) for name, (system, _) in THREE.items() if system != OURS}
    kinds = {name: analysis.edit_kinds("mixed-val", system) for name, (system, _) in THREE.items()}
    position = {name: analysis.by_position(system) for name, (system, _) in THREE.items()}
    script = {name: analysis.wrong_script(system) for name, (system, _) in THREE.items()}
    sources = {name: analysis.ksc2_by_source(system) for name, (system, _) in THREE.items()}
    prompt = analysis.prompt_effect(OURS)
    _f, _axes = plt.subplots(2, 2, figsize=(14, 8.8))
    _axes = _axes.ravel()
    _offsets = {name: (index - 1) * 0.27 for index, name in enumerate(THREE)}
    for _name, (_, _color) in THREE.items():
        _axes[0].bar(np.arange(3) + _offsets[_name], [kinds[_name][k] for k in ("substitutions", "deletions", "insertions")], 0.27, color=_color, label=_name)
    _axes[0].set_xticks(np.arange(3), ["wrong word", "missing word", "extra word"])
    _axes[0].set_ylabel("words")
    _axes[0].set_title(f"Mixed recordings, one pass: kinds of errors\n({kinds['our large']['reference_words']:,} reference words)", fontsize=10)
    _axes[0].legend(fontsize=8)
    _keys = list(position["our large"])
    for _name, (_, _color) in THREE.items():
        _axes[1].bar(np.arange(len(_keys)) + _offsets[_name], [min(130, 100 * position[_name][k]) for k in _keys], 0.27, color=_color)
    _axes[1].set_xticks(np.arange(len(_keys)), [k.replace("part", "utterance") for k in _keys])
    _axes[1].set_ylabel("WER % (cut at 130)")
    _axes[1].set_title("Mixed recordings: errors by place in the recording", fontsize=10)
    for _name, (_, _color) in THREE.items():
        _axes[2].bar(np.arange(3) + _offsets[_name], [100 * script[_name][lang] for lang in LANGUAGES], 0.27, color=_color)
    _axes[2].set_xticks(np.arange(3), [LANGUAGE_NAMES[lang] for lang in LANGUAGES])
    _axes[2].set_ylabel("% of the language's words")
    _axes[2].set_title("Mixed recordings: words written in the other alphabet", fontsize=10)
    _names = sorted(sources["our large"])
    for _name, (_, _color) in THREE.items():
        _axes[3].bar(np.arange(len(_names)) + _offsets[_name], [min(150, 100 * sources[_name][s]["wer"]) for s in _names], 0.27, color=_color)
    _axes[3].set_xticks(np.arange(len(_names)), [f"{s}\n({sources['our large'][s]['units']})" for s in _names], fontsize=8)
    _axes[3].set_ylabel("WER %, language given (cut at 150)")
    _axes[3].set_title("Kazakh KSC2 by source\n(our training saw only 'crowdsourced')", fontsize=10)
    _f.tight_layout()
    _report = selection["report_against_starting_model"]
    log.fig(13, "error_analysis_validation", _f,
            "On the mixed validation recordings (one pass, automatic language) our large model makes fewer errors than "
            + "; than ".join(f"the {name} on {m['better']} of {m['units']} recordings (the same on {m['same']}, more on {m['worse']})" for name, m in moves.items())
            + ". Kinds of errors, starting model → our small → our large: wrong words "
            + " → ".join(f"{kinds[name]['substitutions']:,}" for name in THREE) + ", missing words "
            + " → ".join(f"{kinds[name]['deletions']:,}" for name in THREE) + ", extra words "
            + " → ".join(f"{kinds[name]['insertions']:,}" for name in THREE)
            + ". By place in the recording: " + ", ".join(f"{k.replace('part', 'utterance')} " + " → ".join(f"{100 * position[name][k]:.0f}%" for name in THREE) for k in _keys)
            + ". Words written in the other alphabet: " + ", ".join(f"{LANGUAGE_NAMES[lang]} " + " → ".join(f"{100 * script[name][lang]:.1f}%" for name in THREE) for lang in LANGUAGES)
            + ". Kazakh KSC2 by source: " + ", ".join(f"{s} " + " → ".join(f"{100 * sources[name][s]['wer']:.0f}%" for name in THREE) for s in _names)
            + f". The language our large model detects for a mixed recording is its first utterance's in {100 * prompt['first_language_detected']:.0f}% of cases"
            + (f" (WER {100 * prompt['wer_when_first']:.1f}% then, {100 * prompt['wer_when_other']:.1f}% otherwise)" if prompt["wer_when_other"] is not None else "")
            + "; on single utterances it names the language correctly for "
            + ", ".join(f"{100 * _report[key]['language_accuracy']:.0f}% of {key}" for key in ("fleurs-kk", "fleurs-ru", "fleurs-en")) + ".")
    {"moves": moves, "kinds": kinds, "position": position, "wrong_script": script, "prompt": prompt}
    return


@app.cell
def _(OURS, analysis):
    # The recordings where our large model is worst (validation): read them before trusting an average.
    analysis.worst("mixed-val", OURS, 8)
    return


@app.cell
def _(LANGUAGES, LANGUAGE_NAMES, RESULTS, json, log, np, pd, plt):
    # Step 14 - the final test on test B, drawn from the saved result of the one sanctioned run.
    _path = RESULTS / "test-final-B.json"
    final = json.loads(_path.read_text(encoding="utf-8")) if _path.is_file() else None
    if final:
        _scores = pd.DataFrame(final["scores"])
        _sets = ["mixed", "fleurs-kk", "ksc2-kk", "fleurs-ru", "golos-ru", "fleurs-en"]
        _systems = [("kazakh", "Kazakh turbo (starting model)", "#2a9d8f"), ("cleartalk-large", "ClearTalk large (ours, this round)", "#de693c"),
                    ("turbo", "Whisper large-v3-turbo", "#3b6ea8"), ("cleartalk", "ClearTalk small (ours, round 1)", "#e9b384"),
                    ("small", "Whisper small (untouched)", "#999999"), ("pipeline:auto", "App pipeline, language per turn", "#7b5ea7"),
                    ("pipeline:mixed", "App pipeline, mixed setting", "#444444")]
        _f, _axes = plt.subplots(1, 2, figsize=(18, 4.9), gridspec_kw={"width_ratios": [2.1, 1.1]})

        def _value(system, set_name):
            row = _scores[(_scores.system == system) & (_scores.set == set_name)]
            if not len(row):
                return np.nan
            return 100 * float(row.iloc[0]["wer_auto" if set_name == "mixed" else "wer_forced"])

        _x = np.arange(len(_sets))
        for _i, (_key, _label, _color) in enumerate(_systems):
            # A system that was not measured on a set has no bar (NaN must not be clipped into a full bar).
            _heights = [np.nan if np.isnan(_value(_key, s)) else min(130, _value(_key, s)) for s in _sets]
            _axes[0].bar(_x + (_i - 3) * 0.125, _heights, 0.125, color=_color, label=_label)
        _axes[0].set_xticks(_x, _sets)
        _axes[0].set_ylabel("WER % (mixed: automatic language; others: language given; cut at 130)")
        _axes[0].set_title("Final test on test B, one run: every system on the same frozen audio")
        _axes[0].legend(fontsize=7.5)
        _by = final["mixed_by_language"]
        _names = [k for k, _, _ in _systems if k in _by]
        for _i, _lang in enumerate(LANGUAGES):
            _axes[1].bar(np.arange(len(_names)) + (_i - 1) * 0.27, [min(130, 100 * _by[n][_lang]["wer"]) for n in _names], 0.27,
                         label=LANGUAGE_NAMES[_lang], color={"kk": "#2a9d8f", "ru": "#3b6ea8", "en": "#d08c2b"}[_lang])
        _short = {"kazakh": "starting\nmodel", "cleartalk-large": "ours\nlarge", "turbo": "large-v3\nturbo", "cleartalk": "ours\nsmall",
                  "small": "Whisper\nsmall", "pipeline:auto": "pipeline\nper turn", "pipeline:mixed": "pipeline\nmixed"}
        _axes[1].set_xticks(np.arange(len(_names)), [_short[n] for n in _names], fontsize=7)
        _axes[1].set_title("Mixed test recordings: errors by language", fontsize=10)
        _axes[1].set_ylabel("WER % of the words of each language (cut at 130)")
        _axes[1].legend(fontsize=8)
        _f.tight_layout()
        _cmp = final["cleartalk_large_minus"]

        def _against(other, label):
            return f"against {label}: " + "; ".join(
                f"{s} {_value(other, s):.1f}% → {_value('cleartalk-large', s):.1f}% ({100 * _cmp[other][s]['difference']:+.1f} points, 95% bootstrap "
                f"{100 * _cmp[other][s]['low']:+.1f} to {100 * _cmp[other][s]['high']:+.1f})" for s in _sets if s in _cmp[other])

        log.fig(14, "final_test_b", _f,
                f"Final test on test B, opened once, the same frozen audio for every system ({final['units']} units, 170 per set). Our large model "
                + _against("kazakh", "its untouched starting model") + ". " + _against("cleartalk", "our small model of round 1").capitalize() + ". "
                + _against("turbo", "the untouched large-v3-turbo").capitalize()
                + ". Other points: " + "; ".join(f"{label} " + ", ".join(f"{s} {_value(key, s):.1f}%" for s in _sets if not np.isnan(_value(key, s)))
                                                 for key, label, _ in _systems[4:])
                + ". Reported as measured; the checkpoint, the routes and the recognizer for the mixed setting were fixed on validation before this run.")
    final["cleartalk_large_minus"] if final else "The final test on test B has not been run yet."
    return


@app.cell
def _(RESULTS, json):
    # Step 15 - deployment check written by scripts/check_round2_deployment.py.
    _path = RESULTS / "deployment-check-round2.json"
    json.loads(_path.read_text(encoding="utf-8")) if _path.is_file() else "The deployment check has not been run yet."
    return


if __name__ == "__main__":
    app.run()
