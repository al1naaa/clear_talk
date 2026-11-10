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
    from cleartalk_asr import splits
    from cleartalk_asr.evaluate import RESULTS, eval_tracks
    from cleartalk_asr.paths import EXPORT
    from cleartalk_asr.text import add_counts, edit_counts, pair_counts

    STATE = ROOT / "ml_pipeline_english"
    return EXPORT, RESULTS, ROOT, STATE, add_counts, edit_counts, eval_tracks, guard, json, mo, np, pair_counts, pd, plt, splits


@app.cell
def _(mo):
    mo.md("""
    # Steps 13-16 · Error analysis, final test, deployment, monitoring

    Step 13 uses **validation** only. The final test itself is run once by
    `python -m cleartalk_asr.final_test` (guarded); this notebook only reads its saved transcripts.
    """)
    return


@app.cell
def _(RESULTS, edit_counts, eval_tracks, json, pd):
    def _hypotheses(label):
        saved = json.loads((RESULTS / f"val-{label}-app.json").read_text())
        return {r["track_id"]: r["hypothesis"] for r in saved["tracks"]}

    _frame = eval_tracks("val")
    _base, _tuned = _hypotheses("baseline"), _hypotheses("finetuned")
    _rows = []
    for _r in _frame.itertuples():
        _cb, _ct = edit_counts(_r.text_ls, _base[_r.track_id]), edit_counts(_r.text_ls, _tuned[_r.track_id])
        _rows.append({"track_id": _r.track_id, "kind": _r.kind, "si_sdr_db": _r.si_sdr_db, "snr_db": _r.snr_db,
                      "words": _cb["reference_words"], "reference": _r.text_ls,
                      "pretrained": _base[_r.track_id], "cleartalk": _tuned[_r.track_id],
                      "errors_base": _cb["substitutions"] + _cb["deletions"] + _cb["insertions"],
                      "errors_tuned": _ct["substitutions"] + _ct["deletions"] + _ct["insertions"],
                      **{f"{k}_base": _cb[k] for k in ("substitutions", "deletions", "insertions")},
                      **{f"{k}_tuned": _ct[k] for k in ("substitutions", "deletions", "insertions")}})
    val_tracks = pd.DataFrame(_rows)
    val_tracks["delta"] = val_tracks.errors_tuned - val_tracks.errors_base
    val_tracks.sort_values("delta").head(6)[["reference", "pretrained", "cleartalk", "delta"]]
    return (val_tracks,)


@app.cell
def _(val_tracks):
    # The largest regressions: where the fine-tuned model is worse than the pretrained one.
    val_tracks.sort_values("delta", ascending=False).head(6)[["reference", "pretrained", "cleartalk", "delta", "si_sdr_db"]]
    return


@app.cell
def _(STATE, guard, np, pd, plt, val_tracks):
    _pairs = val_tracks[val_tracks.kind == "pair"].copy()
    _pairs["sdr_bin"] = pd.cut(_pairs.si_sdr_db, [-np.inf, -5, 0, 5, 10, np.inf])
    _by = _pairs.groupby("sdr_bin", observed=True)[["errors_base", "errors_tuned", "words"]].sum()
    _f, _axes = plt.subplots(1, 3, figsize=(16, 3.8))
    _x = np.arange(len(_by))
    _axes[0].bar(_x - 0.2, 100 * _by.errors_base / _by.words, 0.4, color="#999999", label="pretrained")
    _axes[0].bar(_x + 0.2, 100 * _by.errors_tuned / _by.words, 0.4, color="#3b6ea8", label="ClearTalk")
    _axes[0].set_xticks(_x, [str(i) for i in _by.index], rotation=20)
    _axes[0].set_xlabel("separation quality of the track, SI-SDR (dB)")
    _axes[0].set_ylabel("WER (%)")
    _axes[0].legend()
    _axes[0].set_title("By separation quality")
    _kinds = ["substitutions", "deletions", "insertions"]
    _b = [_pairs[f"{k}_base"].sum() for k in _kinds]
    _t = [_pairs[f"{k}_tuned"].sum() for k in _kinds]
    _k = np.arange(3)
    _axes[1].bar(_k - 0.2, _b, 0.4, color="#999999", label="pretrained")
    _axes[1].bar(_k + 0.2, _t, 0.4, color="#3b6ea8", label="ClearTalk")
    _axes[1].set_xticks(_k, _kinds)
    _axes[1].set_ylabel("word errors on validation mixtures")
    _axes[1].set_title("By error type")
    _axes[1].legend()
    _axes[2].hist(_pairs.delta.clip(-25, 25), bins=np.arange(-25.5, 26.5, 1), color="#5a7d9a")
    _axes[2].axvline(0, color="black", linewidth=1)
    _axes[2].set_xlabel("errors per track: ClearTalk - pretrained (negative = better)")
    _axes[2].set_ylabel("tracks")
    _axes[2].set_title("Per-track change")
    _better, _same, _worse = int((_pairs.delta < 0).sum()), int((_pairs.delta == 0).sum()), int((_pairs.delta > 0).sum())
    _low = _by[_by.index.map(lambda b: b.right <= -5)]
    guard.fig(13, "error_analysis_validation", _f,
              f"On {len(_pairs)} separated validation tracks ClearTalk makes fewer errors on {_better}, the same on {_same} and more "
              f"on {_worse}. Substitutions {int(_b[0])} → {int(_t[0])}, deletions {int(_b[1])} → {int(_t[1])}, insertions "
              f"{int(_b[2])} → {int(_t[2])}: the main gain is no longer transcribing the leaked second voice. Tracks where the "
              f"separator failed (SI-SDR ≤ -5 dB) stay at {100 * _low.errors_tuned.sum() / max(_low.words.sum(), 1):.0f}% WER "
              f"({100 * _low.errors_base.sum() / max(_low.words.sum(), 1):.0f}% pretrained): the recognizer cannot recover words "
              "that are not in the audio.",
              state_dir=STATE)
    return


@app.cell
def _(RESULTS, add_counts, edit_counts, json, np, pair_counts, pd, splits):
    final = json.loads((RESULTS / "test-final.json").read_text())
    _frame = pd.DataFrame({k: final[k] for k in ("unit_id", "set", "kind", "reference", "hyp_baseline", "hyp_cleartalk")})

    def _counts(reference, hypothesis, kind="whisper"):
        refs, hyps = reference.split(splits.SEP), hypothesis.split(splits.SEP)
        return pair_counts(refs, hyps, kind) if len(refs) > 1 else add_counts([edit_counts(refs[0], hyps[0], kind)])

    _names = {("v4test", "pair"): "T1 · v4 test mixtures", ("test", "pair"): "T2 · new test mixtures",
              ("test", "single"): "T3 · clean single speaker"}
    _rows, test_units = [], []
    for _key, _name in _names.items():
        _group = _frame[(_frame.set == _key[0]) & (_frame.kind == _key[1])]
        _b = [_counts(r, h) for r, h in zip(_group.reference, _group.hyp_baseline)]
        _t = [_counts(r, h) for r, h in zip(_group.reference, _group.hyp_cleartalk)]
        _bl = add_counts(_counts(r, h, "legacy") for r, h in zip(_group.reference, _group.hyp_baseline))
        _tl = add_counts(_counts(r, h, "legacy") for r, h in zip(_group.reference, _group.hyp_cleartalk))
        _eb, _et = np.array([c["errors"] for c in _b]), np.array([c["errors"] for c in _t])
        _w = np.array([c["reference_words"] for c in _b])
        _rng = np.random.default_rng(0)
        _diffs = []
        for _ in range(2000):
            _i = _rng.integers(0, len(_w), len(_w))
            _diffs.append((_et[_i].sum() - _eb[_i].sum()) / _w[_i].sum())
        _sb, _st = add_counts(_b), add_counts(_t)
        _rows.append({"set": _name, "units": len(_group), "reference_words": int(_w.sum()),
                      "pretrained_wer": _sb["wer"], "cleartalk_wer": _st["wer"],
                      "pretrained_errors": _sb["errors"], "cleartalk_errors": _st["errors"],
                      "pretrained_sdi": [_sb["substitutions"], _sb["deletions"], _sb["insertions"]],
                      "cleartalk_sdi": [_st["substitutions"], _st["deletions"], _st["insertions"]],
                      "difference": _st["wer"] - _sb["wer"], "ci_low": float(np.percentile(_diffs, 2.5)),
                      "ci_high": float(np.percentile(_diffs, 97.5)),
                      "relative_change": _st["wer"] / _sb["wer"] - 1,
                      "pretrained_wer_legacy": _bl["wer"], "cleartalk_wer_legacy": _tl["wer"]})
    test_table = pd.DataFrame(_rows)
    (RESULTS / "final-summary.json").write_text(json.dumps(
        {"test": test_table.to_dict("records"), "cleartalk_wer_all_units": final["cleartalk_wer_all_units"]}, indent=2) + "\n")
    test_table[["set", "units", "reference_words", "pretrained_wer", "cleartalk_wer", "difference", "ci_low", "ci_high"]]
    return (test_table,)


@app.cell
def _(STATE, guard, np, plt, test_table):
    _f, _ax = plt.subplots(figsize=(8.5, 3.8))
    _x = np.arange(len(test_table))
    _ax.bar(_x - 0.2, 100 * test_table.pretrained_wer, 0.4, color="#999999", label="pretrained Whisper small.en")
    _ax.bar(_x + 0.2, 100 * test_table.cleartalk_wer, 0.4, color="#3b6ea8", label="ClearTalk fine-tuned")
    for _i, _r in test_table.iterrows():
        _ax.text(_i - 0.2, 100 * _r.pretrained_wer, f"{100 * _r.pretrained_wer:.1f}", ha="center", va="bottom", fontsize=9)
        _ax.text(_i + 0.2, 100 * _r.cleartalk_wer, f"{100 * _r.cleartalk_wer:.1f}", ha="center", va="bottom", fontsize=9)
    _ax.set_xticks(_x, [s.replace(" · ", "\n") for s in test_table.set])
    _ax.set_ylabel("WER (%) · app decoding")
    _ax.legend()
    _ax.set_title("Step 14 · final test, scored once on unseen speakers")
    guard.fig(14, "final_test", _f,
              "; ".join(f"{r.set}: {100 * r.pretrained_wer:.2f}% → {100 * r.cleartalk_wer:.2f}% WER over {r.reference_words:,} words "
                        f"(difference {100 * r.difference:+.2f} points, 95% bootstrap {100 * r.ci_low:+.2f} to {100 * r.ci_high:+.2f})"
                        for r in test_table.itertuples())
              + ". Same frozen audio for both recognizers, reported as measured; the model was fixed on validation before this run.",
              state_dir=STATE)
    return


@app.cell
def _(EXPORT, RESULTS, ROOT, eval_tracks, json):
    # Step 15: the deployed artifact, loaded the way the service loads it, reproduces the validated transcripts.
    import sys as _sys

    _sys.path.insert(0, str(ROOT))
    from backend.config import Settings as _Settings
    from backend.pipeline import AudioPipeline as _Pipeline
    from backend.transcription import transcribe_path as _transcribe

    _settings = _Settings()
    _pipeline = _Pipeline(_settings)
    _model = _pipeline._load_asr("cleartalk")  # verifies the profile's SHA-256 signatures first
    _saved = {r["track_id"]: r["hypothesis"] for r in json.loads((RESULTS / "val-finetuned-app.json").read_text())["tracks"]}
    _sample = eval_tracks("val").iloc[::45]
    _same = sum(_transcribe(_model, ROOT / r.path)["text"] == _saved[r.track_id] for r in _sample.itertuples())
    deployment_check = {"default_model": _settings.default_model, "export": str(EXPORT.relative_to(ROOT)),
                        "tracks_checked": len(_sample), "identical_transcripts": int(_same)}
    (RESULTS / "deployment-check.json").write_text(json.dumps(deployment_check, indent=2) + "\n")
    deployment_check
    return (deployment_check,)


@app.cell
def _(EXPORT, json, test_table):
    # The model card shown in the app gets the measured test numbers.
    _card_path = EXPORT / "model_card.json"
    _card = json.loads(_card_path.read_text())
    _rows = {r["set"][:2]: r for r in test_table.to_dict("records")}
    _card["test"] = {"baseline_wer": _rows["T2"]["pretrained_wer"], "cleartalk_wer": _rows["T2"]["cleartalk_wer"],
                     "mixtures": _rows["T2"]["units"], "reference_words": _rows["T2"]["reference_words"]}
    _card["test_v4"] = {"baseline_wer": _rows["T1"]["pretrained_wer"], "cleartalk_wer": _rows["T1"]["cleartalk_wer"],
                        "mixtures": _rows["T1"]["units"], "reference_words": _rows["T1"]["reference_words"]}
    _card["test_clean"] = {"baseline_wer": _rows["T3"]["pretrained_wer"], "cleartalk_wer": _rows["T3"]["cleartalk_wer"],
                           "utterances": _rows["T3"]["units"], "reference_words": _rows["T3"]["reference_words"]}
    _card_path.write_text(json.dumps(_card, indent=2) + "\n")
    _card["test"]
    return


if __name__ == "__main__":
    app.run()
