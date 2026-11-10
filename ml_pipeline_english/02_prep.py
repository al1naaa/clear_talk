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
    from cleartalk_asr import dataset, splits
    from cleartalk_asr.mixtures import PLANS, load_tracks
    from cleartalk_asr.text import edit_counts

    STATE = ROOT / "ml_pipeline_english"
    # Step 5 rule, fixed from the validation baseline below: a separated track whose
    # SI-SDR is under this value is dropped from TRAINING targets (never from evaluation).
    SI_SDR_MIN = -5.0
    return (PLANS, ROOT, SI_SDR_MIN, STATE, dataset, edit_counts, guard, json, load_tracks, mo, np, pd, plt,
            splits)


@app.cell
def _(mo):
    mo.md("""
    # Steps 4-8 · Cleaning, data engineering, split, features, preprocessing

    Each separated track is paired with the transcript of the voice it contains. Clean
    references are used only for that pairing and for the cleaning rule below.
    """)
    return


@app.cell
def _(PLANS, STATE, guard, json, plt):
    report = json.loads((PLANS / "cleaning_report.json").read_text())
    _labels = ["all train-clean-100", "minus demo speakers", "has punctuated text",
               "punctuated words = official", "1-16.5 s long"]
    _values = [report["train_total"], report["after_excluding_demo_speakers"],
               report["after_requiring_punctuated_text"], report["after_requiring_pc_words_equal_official"],
               report["after_duration_1_to_16.5s"]]
    _f, _ax = plt.subplots(figsize=(8, 3.4))
    _ax.barh(_labels[::-1], _values[::-1], color="#3b6ea8")
    for _y, _v in enumerate(_values[::-1]):
        _ax.text(_v, _y, f" {_v:,}", va="center", fontsize=8)
    _ax.set_xlabel("training utterances kept")
    _ax.set_title("Step 4 · cleaning funnel (training pool only)")
    guard.fig(4, "cleaning_funnel", _f,
              f"{report['train_total']:,} training utterances → {report['after_duration_1_to_16.5s']:,} usable "
              f"({100 * report['after_duration_1_to_16.5s'] / report['train_total']:.1f}%). The biggest cuts are missing "
              "punctuated text and punctuated text whose words differ from the official transcript (e.g. 'misses' for "
              "'missus', dropped phrases); validation/test utterances are not filtered.",
              state_dir=STATE)
    report
    return (report,)


@app.cell
def _(ROOT, edit_counts, json, load_tracks, np, pd):
    # Validation baseline (pretrained Whisper, app decoding): how recognizable is a track
    # as a function of its separation quality?
    _val = load_tracks("val")
    _res = json.loads((ROOT / "data/asr/results/val-baseline-app.json").read_text())
    _hyp = {r["track_id"]: r["hypothesis"] for r in _res["tracks"]}
    _pairs = _val[_val.kind == "pair"].copy()
    _counts = [edit_counts(r, _hyp[t]) for r, t in zip(_pairs.text_ls, _pairs.track_id)]
    _pairs["errors"] = [c["substitutions"] + c["deletions"] + c["insertions"] for c in _counts]
    _pairs["words"] = [c["reference_words"] for c in _counts]
    _pairs["bin"] = pd.cut(_pairs.si_sdr_db, [-np.inf, -10, -5, 0, 5, 10, np.inf])
    val_by_sdr = _pairs.groupby("bin", observed=True).agg(tracks=("errors", "size"), errors=("errors", "sum"),
                                                          words=("words", "sum"))
    val_by_sdr["wer"] = val_by_sdr.errors / val_by_sdr.words
    val_by_sdr
    return (val_by_sdr,)


@app.cell
def _(SI_SDR_MIN, STATE, guard, load_tracks, np, plt, val_by_sdr):
    train_tracks_all = load_tracks("train")
    _pairs = train_tracks_all[train_tracks_all.kind == "pair"]
    _f, _axes = plt.subplots(1, 2, figsize=(12, 3.6))
    _axes[0].hist(_pairs.si_sdr_db.clip(-30, 30), bins=60, color="#3b6ea8")
    _axes[0].axvline(SI_SDR_MIN, color="#9b4d4d", linestyle="--")
    _axes[0].set_xlabel("SI-SDR of the separated track vs. its clean voice (dB)")
    _axes[0].set_ylabel("training tracks")
    _axes[0].set_title("Separation quality of training tracks")
    _axes[1].bar([str(b) for b in val_by_sdr.index], 100 * val_by_sdr.wer, color="#d08c2b")
    _axes[1].set_ylabel("pretrained Whisper WER (%)")
    _axes[1].set_xlabel("SI-SDR bin (validation tracks)")
    _axes[1].set_title("Below -5 dB the words are not recoverable")
    _axes[1].tick_params(axis="x", rotation=25)
    _dropped = float((_pairs.si_sdr_db < SI_SDR_MIN).mean())
    _low = val_by_sdr[val_by_sdr.index.map(lambda b: b.right <= SI_SDR_MIN)]
    guard.fig(5, "separation_quality_and_filter", _f,
              f"{100 * _dropped:.1f}% of {len(_pairs):,} separated training tracks fall below {SI_SDR_MIN:g} dB. On validation "
              f"the pretrained recognizer already has {100 * _low.errors.sum() / max(_low.words.sum(), 1):.0f}% WER on such "
              "tracks: their transcript is not in the audio (the separator failed), so training on them would teach the "
              "model to invent words. They are dropped from training targets only; evaluation keeps every track.",
              state_dir=STATE)
    return (train_tracks_all,)


@app.cell
def _(SI_SDR_MIN, STATE, guard, np, plt, splits, train_tracks_all):
    train_tracks = train_tracks_all[(train_tracks_all.kind == "single") | (train_tracks_all.si_sdr_db >= SI_SDR_MIN)]
    train_tracks = train_tracks.reset_index(drop=True)
    _val = splits.eval_frame(splits.VAL_SETS)
    _test = splits.eval_frame(splits.TEST_SETS)
    overlap = splits.check_disjoint(train_tracks, _val, _test)
    frozen = splits.freeze_test(_test)
    val_units, test_count = _val, len(_test)
    _f, _axes = plt.subplots(1, 2, figsize=(12, 3.6))
    _bins = np.arange(0, 21, 1)
    _axes[0].hist(train_tracks.duration, bins=_bins, density=True, histtype="step", linewidth=2, color="#3b6ea8",
                  label="training tracks (after re-weighting)")
    _all_train = __import__("pandas").read_csv(splits.PIPELINE.parent / "data/asr/utterances.csv.gz")
    _axes[0].hist(_all_train[_all_train.role == "train"].duration.clip(upper=20), bins=_bins, density=True,
                  histtype="step", linewidth=1, linestyle="--", color="grey", label="raw train-clean-100")
    _axes[0].hist(_all_train[_all_train.role != "train"].duration.clip(upper=20), bins=_bins, density=True,
                  histtype="step", linewidth=2, color="#5a9e5a", label="validation + test utterances")
    _axes[0].set_xlabel("utterance duration (s)")
    _axes[0].legend(fontsize=8)
    _axes[0].set_title("Duration re-weighting")
    _sizes = {"train tracks": len(train_tracks), "val units": len(_val), "test units": len(_test)}
    _axes[1].bar(list(_sizes), list(_sizes.values()), color=["#3b6ea8", "#d08c2b", "#5a9e5a"])
    for _x, _v in enumerate(_sizes.values()):
        _axes[1].text(_x, _v, f"{_v:,}", ha="center", va="bottom", fontsize=9)
    _axes[1].set_title("Step 6 · split sizes (speaker-disjoint)")
    guard.fig(6, "split_and_duration_matching", _f,
              f"{len(train_tracks):,} training tracks, {len(_val):,} validation units and {len(_test):,} frozen test units "
              f"(mixtures or single utterances). Speaker overlap between roles: {overlap}. Re-weighting moved the training "
              f"median duration to {train_tracks.duration.median():.1f} s (raw corpus 14.0 s) toward validation/test.",
              state_dir=STATE)
    {"overlap": overlap, "frozen_test": frozen}
    return train_tracks, val_units


@app.cell
def _(STATE, dataset, guard, np, plt, train_tracks):
    _row = train_tracks[(train_tracks.kind == "pair") & train_tracks.si_sdr_db.between(3, 6)].iloc[0]
    _source = dataset.features(dataset.load_audio(_row.source_path))
    _separated = dataset.features(dataset.load_audio(_row.path))
    _frames = int(min(1500, 100 * (_row.mixture_seconds + 0.5)))
    _f, _axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    _axes[0].imshow(_source[:, :_frames], origin="lower", aspect="auto", cmap="magma")
    _axes[0].set_title(f"Clean utterance {_row.utt_id} (Whisper log-mel, 80 bands)")
    _axes[1].imshow(_separated[:, :_frames], origin="lower", aspect="auto", cmap="magma")
    _axes[1].set_title(f"Same voice after mixing + noise + SepFormer (SI-SDR {_row.si_sdr_db:.1f} dB, 8 kHz)")
    _axes[1].set_xlabel("frames (10 ms)")
    _cut = float(np.mean(_separated[60:, :_frames]) - np.mean(_source[60:, :_frames]))
    guard.fig(7, "logmel_clean_vs_separated", _f,
              "Whisper's input features are fixed log-mel spectrograms (no fitted statistics, so nothing can leak from "
              "validation/test). The separated track loses everything above 4 kHz (top quarter of the bands is empty, "
              f"mean level change {_cut:+.2f}) and carries residual interference: this is the domain shift the fine-tuning "
              "has to absorb.",
              state_dir=STATE)
    return


@app.cell
def _(STATE, dataset, guard, np, plt, train_tracks):
    _sample = train_tracks.sample(min(3000, len(train_tracks)), random_state=0)
    _lengths = [len(dataset.target_tokens(t)) for t in _sample.text]
    _ts = np.mean([dataset.uses_timestamps(t) for t in train_tracks.track_id])
    _f, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.hist(_lengths, bins=40, color="#5a7d9a")
    _ax.axvline(dataset.MAX_LABEL_TOKENS, color="#9b4d4d", linestyle="--")
    _ax.set_xlabel("target length (Whisper tokens incl. special tokens)")
    _ax.set_title("Step 8 · target sequences")
    guard.fig(8, "target_token_lengths", _f,
              f"Targets are Whisper BPE tokens of the punctuated transcript: median {int(np.median(_lengths))}, max "
              f"{max(_lengths)} tokens, far below the {dataset.MAX_LABEL_TOKENS}-token limit, so nothing is truncated. "
              f"{100 * _ts:.0f}% of training tracks keep segment timestamps so the app's timed segments still work.",
              state_dir=STATE)
    return


if __name__ == "__main__":
    app.run()
