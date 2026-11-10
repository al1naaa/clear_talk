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
    import os

    os.environ["CLEARTALK_BASE"] = "kazakh-turbo"
    from cleartalk_ml import noise, round4
    from cleartalk_ml import pipeline_log as log
    from cleartalk_ml.paths import RESULTS

    log.use(round4.R4)
    return RESULTS, json, log, mo, noise, np, pd, plt, round4


@app.cell
def _(mo):
    mo.md("""
    # Round 4 · Steps 1-8: why and how the large model is trained in noise

    No new speech data. New: the robustness check on validation (why), the noise added to training examples (how),
    and test D. No model is fitted here.
    """)
    return


@app.cell
def _(RESULTS, json, log, np, pd, plt):
    # Step 2 - the app on validation recordings in noise (cleartalk_ml.robustness, per-turn setting).
    robust = pd.DataFrame(json.loads((RESULTS / "robustness.json").read_text(encoding="utf-8")))
    auto = robust[robust.setting == "auto"]
    order = ["clean", "white-20", "white-10", "babble-10", "babble-5"]
    sets = ["mixed", "short-kk", "short-ru", "short-en", "long"]
    _f, _ax = plt.subplots(figsize=(12, 4.2))
    _x = np.arange(len(order))
    for _i, _set in enumerate(sets):
        _vals = [float(auto[(auto.condition == c) & (auto.set == _set)].wer.iloc[0]) * 100 if len(auto[(auto.condition == c) & (auto.set == _set)]) else np.nan for c in order]
        _ax.bar(_x + (_i - 2) * 0.16, _vals, 0.16, label=_set)
    _ax.set_xticks(_x, order)
    _ax.set_ylabel("WER %, through the app (validation)")
    _ax.set_title("The app in noise: clean, white noise, babble (other people talking); SNR in dB")
    _ax.legend(fontsize=8)
    _f.tight_layout()
    _get = lambda c, s: 100 * float(auto[(auto.condition == c) & (auto.set == s)].wer.iloc[0])  # noqa: E731
    log.fig(2, "app_in_noise_validation", _f,
            "The app (language detected per turn, both decoding options on) on validation recordings with added noise. Mixed recordings: "
            + ", ".join(f"{c} {_get(c, 'mixed'):.1f}%" for c in order)
            + f"; 7-minute conversations: clean {_get('clean', 'long'):.1f}%, white 10 dB {_get('white-10', 'long'):.1f}%, babble 10 dB {_get('babble-10', 'long'):.1f}%. "
            "In babble most of the added errors are extra words (the background voices are written down); in white noise they are wrong words. "
            "Short phrases are the weakest already when clean (Kazakh and Russian about 19-21%): mostly names and joined words, and a few "
            "Kazakh phrases detected as English with confidence.")
    auto.pivot_table(index="set", columns="condition", values="wer").mul(100).round(1)
    return


@app.cell
def _(log, noise, np, plt):
    # Steps 5 and 7 - what a noisy training example looks like to the model (128-band log-mel), and how often noise is added.
    from cleartalk_ml.dataset import features
    from cleartalk_ml.splits import training_examples

    examples = sorted(training_examples(), key=lambda e: e["id"])
    specs = [noise.spec_for(e["id"], 0.5) for e in examples]
    talkers = noise.talker_pool()
    example = next(e for e in examples if e["kind"] == "mixed")
    from cleartalk_ml.dataset import example_audio

    clean = example_audio(example)
    views = {"clean": clean, "white noise 10 dB": noise.add(clean, {"kind": "white", "snr": 10.0, "seed": 1}, talkers),
             "babble 10 dB": noise.add(clean, {"kind": "babble", "snr": 10.0, "seed": 2}, talkers)}
    _f, _axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    for _ax, (_name, _audio) in zip(_axes, views.items()):
        _ax.imshow(features(_audio)[:, :int(len(_audio) / 160)], origin="lower", aspect="auto", cmap="magma")
        _ax.set_title(_name, fontsize=10)
        _ax.set_ylabel("mel band")
    _axes[-1].set_xlabel("frame (10 ms)")
    _f.tight_layout()
    _kinds = {k: sum(1 for s in specs if s and s["kind"] == k) for k in noise.KINDS}
    log.fig(7, "noisy_training_example", _f,
            f"One mixed training recording as the model sees it: clean, with white noise and with babble at 10 dB SNR. Of the {len(examples):,} "
            f"training examples {sum(s is not None for s in specs):,} get noise ({', '.join(f'{k} {n:,}' for k, n in _kinds.items())}); SNR uniform "
            "in 0-20 dB (white, pink) or 3-20 dB (babble of 3-5 talkers). Babble comes from 3,000 training utterances; the validation noise "
            "and the test noise use other talkers. The target is always the transcript of the main speech.")
    _kinds
    return


if __name__ == "__main__":
    app.run()
