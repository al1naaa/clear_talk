import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from collections import Counter
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
    from cleartalk_asr.corpus import build_manifest
    from cleartalk_asr.text import edit_counts, normalize

    STATE = ROOT / "ml_pipeline_english"
    return Counter, ROOT, STATE, build_manifest, edit_counts, guard, json, mo, normalize, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Step 1-3 · Data inspection, EDA and the prediction problem

    **Unit of data:** one LibriSpeech utterance (a read-aloud sentence or paragraph) with its
    reference transcript. Two utterances from different speakers are later mixed with noise,
    separated by the frozen SepFormer, and each separated voice becomes one training example.
    """)
    return


@app.cell
def _(ROOT, build_manifest, pd):
    cache = ROOT / "data/asr/utterances.csv.gz"
    if cache.is_file():
        utterances = pd.read_csv(cache, dtype={"speaker": str, "chapter": str})
    else:
        utterances = build_manifest()
        utterances.to_csv(cache, index=False)
    utterances.groupby(["role", "subset"]).agg(
        utterances=("utt_id", "size"), speakers=("speaker", "nunique"),
        hours=("duration", lambda s: round(s.sum() / 3600, 2)),
        pc_coverage=("has_pc", "mean"))
    return (utterances,)


@app.cell
def _(STATE, guard, utterances):
    profile = guard.profile(utterances, target="text", group_col="speaker", state_dir=STATE)
    {"shape": profile["shape"], "duplicates": profile["duplicates"],
     "leakage_suspects": profile["leakage_suspects"], "traits": profile["traits"]}
    return (profile,)


@app.cell
def _(utterances):
    # Leakage checks that matter for speech: the same voice or the same sentence on both sides.
    speakers = utterances.groupby("role")["speaker"].apply(set)
    speaker_overlap = {f"{a}&{b}": len(speakers[a] & speakers[b])
                       for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    texts = utterances.assign(key=utterances["text_ls"].str.strip()).groupby("role")["key"].apply(set)
    text_overlap = {f"{a}&{b}": len(texts[a] & texts[b])
                    for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(v == 0 for v in speaker_overlap.values()), speaker_overlap
    {"speaker_overlap": speaker_overlap, "identical_sentence_overlap": text_overlap}
    return speaker_overlap, text_overlap


@app.cell
def _(STATE, guard, plt, utterances):
    _summary = utterances.groupby("subset").agg(hours=("duration", lambda s: s.sum() / 3600),
                                                speakers=("speaker", "nunique"))
    _order = ["train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"]
    _summary = _summary.loc[_order]
    _f, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
    _colors = ["#3b6ea8"] + ["#d08c2b"] * 2 + ["#5a9e5a"] * 2
    _axes[0].bar(_order, _summary["hours"], color=_colors)
    _axes[0].set_ylabel("hours of speech")
    _axes[0].set_title("Audio per subset (blue train, orange validation, green test)")
    _axes[1].bar(_order, _summary["speakers"], color=_colors)
    _axes[1].set_ylabel("distinct speakers")
    _axes[1].set_title("Speakers per subset")
    for _ax in _axes:
        _ax.tick_params(axis="x", rotation=25)
    guard.fig(1, "corpus_overview", _f,
              f"Training has {_summary.loc['train-clean-100','hours']:.1f} h from "
              f"{int(_summary.loc['train-clean-100','speakers'])} speakers; validation "
              f"{_summary.loc[['dev-clean','dev-other'],'hours'].sum():.1f} h and test "
              f"{_summary.loc[['test-clean','test-other'],'hours'].sum():.1f} h use different speakers. "
              "dev-other is only a 528-utterance local subset, so validation leans on dev-clean.",
              state_dir=STATE)
    return


@app.cell
def _(STATE, guard, plt, utterances):
    _missing = 1 - utterances.groupby("subset")["has_pc"].mean()
    _f, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.barh(_missing.index, 100 * _missing.to_numpy(), color="#9b4d4d")
    _ax.set_xlabel("% utterances without a punctuated (LibriSpeech-PC) transcript")
    _ax.set_title("Missing punctuated targets")
    guard.fig(1, "missing_punctuated_text", _f,
              f"{100 * (1 - utterances['has_pc'].mean()):.1f}% of utterances have no punctuated transcript "
              f"(worst: {_missing.idxmax()} at {100 * _missing.max():.1f}%). These cannot be training targets "
              "without losing punctuation, but they keep their official text for scoring.",
              state_dir=STATE)
    return


@app.cell
def _(STATE, guard, np, plt, utterances):
    _f, _ax = plt.subplots(figsize=(8, 3.6))
    _bins = np.arange(0, 36, 1)
    for _role, _color in (("train", "#3b6ea8"), ("val", "#d08c2b"), ("test", "#5a9e5a")):
        _d = utterances.loc[utterances.role == _role, "duration"]
        _ax.hist(_d, bins=_bins, density=True, histtype="step", linewidth=2, color=_color, label=_role)
    _ax.axvline(30, color="black", linestyle="--", linewidth=1)
    _ax.text(29.5, _ax.get_ylim()[1] * 0.9, "Whisper 30 s window", ha="right", fontsize=8)
    _ax.set_xlabel("utterance duration (s)")
    _ax.set_ylabel("density")
    _ax.legend()
    _ax.set_title("Utterance duration by role")
    _train = utterances.loc[utterances.role == "train", "duration"]
    _eval = utterances.loc[utterances.role != "train", "duration"]
    guard.fig(2, "duration_by_role", _f,
              f"Training utterances are long (median {_train.median():.1f} s, {100 * (_train > 16).mean():.0f}% "
              f"over 16 s) while validation/test are shorter (median {_eval.median():.1f} s). A two-speaker "
              "mixture must fit Whisper's 30 s window, so training sources are capped and short ones kept.",
              state_dir=STATE)
    return


@app.cell
def _(STATE, guard, plt, utterances):
    _f, _axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for _role, _color in (("train", "#3b6ea8"), ("val", "#d08c2b"), ("test", "#5a9e5a")):
        _part = utterances[utterances.role == _role]
        _axes[0].hist(_part["n_words"], bins=40, density=True, histtype="step", linewidth=2,
                      color=_color, label=_role)
        _axes[1].hist(_part["words_per_second"], bins=40, density=True, histtype="step", linewidth=2,
                      color=_color, label=_role)
    _axes[0].set_xlabel("words per utterance")
    _axes[1].set_xlabel("speaking rate (words / s)")
    _axes[0].legend()
    _axes[0].set_title("Transcript length")
    _axes[1].set_title("Speaking rate")
    _rates = utterances.groupby("role")["words_per_second"].median()
    guard.fig(2, "length_and_speaking_rate", _f,
              f"Median speaking rate is {_rates['train']:.2f} words/s in train, {_rates['val']:.2f} in "
              f"validation and {_rates['test']:.2f} in test: the reading pace is similar, so the main "
              "difference between roles is length, not speaking style.",
              state_dir=STATE)
    return


@app.cell
def _(STATE, guard, plt, utterances):
    _g = (utterances.drop_duplicates(["role", "speaker"]).groupby(["role", "gender"]).size()
          .unstack(fill_value=0).loc[["train", "val", "test"]])
    _f, _ax = plt.subplots(figsize=(6, 3.2))
    _g.plot(kind="bar", stacked=True, ax=_ax, color={"F": "#c25b8a", "M": "#4f7cb8"})
    _ax.set_ylabel("speakers")
    _ax.set_title("Speaker gender per role")
    _ax.tick_params(axis="x", rotation=0)
    _share = (_g["F"] / _g.sum(axis=1)).round(2).to_dict()
    guard.fig(2, "gender_balance", _f,
              f"Female share of speakers is {_share}. The roles are balanced, so a gender skew "
              "cannot explain a validation/test difference.",
              state_dir=STATE)
    return


@app.cell
def _(Counter, STATE, guard, normalize, np, plt, utterances):
    def _words(frame):
        _c = Counter()
        for _t in frame["text_ls"]:
            _c.update(normalize(_t, "legacy").split())
        return _c

    _train_vocab = _words(utterances[utterances.role == "train"])
    _oov = {}
    for _role in ("val", "test"):
        _c = _words(utterances[utterances.role == _role])
        _oov[_role] = sum(n for w, n in _c.items() if w not in _train_vocab) / sum(_c.values())
    _freq = np.array(sorted(_train_vocab.values(), reverse=True))
    _f, _axes = plt.subplots(1, 2, figsize=(11, 3.5))
    _axes[0].loglog(np.arange(1, len(_freq) + 1), _freq, color="#3b6ea8")
    _axes[0].set_xlabel("word rank")
    _axes[0].set_ylabel("count in training transcripts")
    _axes[0].set_title("Zipf's law in the training text")
    _axes[1].bar(list(_oov), [100 * v for v in _oov.values()], color=["#d08c2b", "#5a9e5a"])
    _axes[1].set_ylabel("% running words unseen in training")
    _axes[1].set_title("Out-of-vocabulary rate")
    guard.fig(2, "vocabulary_and_oov", _f,
              f"The training text has {len(_train_vocab):,} word types with a Zipfian tail; "
              f"{100 * _oov['val']:.2f}% of validation and {100 * _oov['test']:.2f}% of test words never occur "
              "in training. Those words must come from Whisper's pretrained knowledge, which is one reason to "
              "adapt Whisper gently (LoRA) instead of retraining it.",
              state_dir=STATE)
    return


@app.cell
def _(ROOT, edit_counts, json, pd):
    # Error profile of the CURRENT app (SepFormer -> Whisper small.en) on the v4 VALIDATION mixtures.
    # Test results are deliberately not opened here.
    _results = json.loads((ROOT / "archive/evaluation/v4/asr_validation/baseline/results.json").read_text())
    _recipe = {r["id"]: r for r in json.loads((ROOT / "data/v4/asr_val_manifest.json").read_text())["pairs"]}
    _rows = []
    for _e in _results["examples"]:
        for _i, _j in enumerate(_e["permutation"]):
            _ref, _hyp = _e["references"][_i], _e["hypotheses"][_j]
            _w, _l = edit_counts(_ref, _hyp, "whisper"), edit_counts(_ref, _hyp, "legacy")
            _r = _recipe[_e["id"]]
            _rows.append({"id": _e["id"], "reference": _ref, "hypothesis": _hyp,
                          "si_sdr_db": _e["baseline_si_sdr_db"], "snr_db": _r["snr_db"],
                          "noise": _r["noise"], "delay": _r["delay_fraction"], "reverb": _r["reverb"],
                          **{f"w_{k}": v for k, v in _w.items()}, **{f"l_{k}": v for k, v in _l.items()}})
    val_errors = pd.DataFrame(_rows)
    _tot = val_errors.sum(numeric_only=True)
    baseline_val_summary = {
        "tracks": len(val_errors),
        "wer_whisper_norm": (_tot.w_substitutions + _tot.w_deletions + _tot.w_insertions) / _tot.w_reference_words,
        "wer_legacy_norm": (_tot.l_substitutions + _tot.l_deletions + _tot.l_insertions) / _tot.l_reference_words,
    }
    baseline_val_summary
    return baseline_val_summary, val_errors


@app.cell
def _(STATE, baseline_val_summary, guard, np, plt, val_errors):
    _v = val_errors.assign(errors=val_errors.w_substitutions + val_errors.w_deletions + val_errors.w_insertions)
    _f, _axes = plt.subplots(1, 3, figsize=(14, 3.6))
    _per = _v.groupby("id").agg(errors=("errors", "sum"), words=("w_reference_words", "sum"),
                                sdr=("si_sdr_db", "first"))
    _axes[0].scatter(_per["sdr"], 100 * _per["errors"] / _per["words"], s=18, color="#3b6ea8")
    _axes[0].set_xlabel("separation quality, SI-SDR (dB)")
    _axes[0].set_ylabel("WER per mixture (%)")
    _axes[0].set_title("Worse separation → more word errors")
    _by_snr = _v.groupby("snr_db")[["errors", "w_reference_words"]].sum()
    _axes[1].bar(_by_snr.index.astype(str), 100 * _by_snr.errors / _by_snr.w_reference_words, color="#d08c2b")
    _axes[1].set_xlabel("target noise level, SNR (dB)")
    _axes[1].set_ylabel("WER (%)")
    _axes[1].set_title("WER by noise level")
    _sdi = _v[["w_substitutions", "w_deletions", "w_insertions"]].sum()
    _axes[2].bar(["substitutions", "deletions", "insertions"], _sdi.to_numpy(), color="#9b4d4d")
    _axes[2].set_title("Error types (Whisper normalizer)")
    _corr = float(np.corrcoef(_per["sdr"], _per["errors"] / _per["words"])[0, 1])
    guard.fig(2, "baseline_error_profile_validation", _f,
              f"On the 48 v4 validation mixtures the current app has {100 * baseline_val_summary['wer_whisper_norm']:.1f}% "
              f"WER with Whisper normalization ({100 * baseline_val_summary['wer_legacy_norm']:.1f}% with the old rule). "
              f"Errors correlate with separation quality (r = {_corr:.2f}) and are mostly substitutions "
              f"({int(_sdi.w_substitutions)} vs {int(_sdi.w_deletions)} deletions, {int(_sdi.w_insertions)} insertions): "
              "the recognizer mishears artefact-laden speech, which is what fine-tuning on separated audio targets.",
              state_dir=STATE)
    return


@app.cell
def _(Counter, STATE, guard, normalize, plt, val_errors):
    from jiwer import process_words as _process

    _pairs = Counter()
    for _ref, _hyp in zip(val_errors.reference, val_errors.hypothesis):
        _r, _h = normalize(_ref), normalize(_hyp)
        if not _r or not _h:
            continue
        _out = _process(_r, _h)
        _rw, _hw = _r.split(), _h.split()
        for _chunk in _out.alignments[0]:
            if _chunk.type == "substitute":
                for _a, _b in zip(_rw[_chunk.ref_start_idx:_chunk.ref_end_idx],
                                  _hw[_chunk.hyp_start_idx:_chunk.hyp_end_idx]):
                    _pairs[f"{_a} → {_b}"] += 1
    _top = _pairs.most_common(15)
    _f, _ax = plt.subplots(figsize=(7, 4.2))
    _ax.barh([p for p, _ in _top][::-1], [n for _, n in _top][::-1], color="#5a7d9a")
    _ax.set_xlabel("count on validation")
    _ax.set_title("Most frequent word substitutions (reference → recognized)")
    guard.fig(2, "top_substitutions_validation", _f,
              f"{sum(_pairs.values())} substituted words fall into {len(_pairs)} distinct pairs and the top 15 "
              f"cover only {100 * sum(n for _, n in _top) / max(sum(_pairs.values()), 1):.0f}%: errors are spread "
              "across acoustically similar words, not one systematic spelling problem a text rule could fix.",
              state_dir=STATE)
    return


@app.cell
def _(baseline_val_summary, mo, speaker_overlap, text_overlap, utterances):
    _counts = utterances.groupby("role").size().to_dict()
    mo.md(f"""
    ## Findings (steps 1-2)

    * {len(utterances):,} utterances: {_counts}. Speaker overlap between roles: {speaker_overlap} - **none**.
    * Identical sentences appearing in two roles: {text_overlap}. They are common phrases read by different
      people; with disjoint speakers they do not leak audio.
    * Current app on v4 validation: **{100 * baseline_val_summary['wer_whisper_norm']:.1f}% WER** (Whisper normalizer).
      Errors grow as separation quality falls → adapt the recognizer to separated speech.
    * Punctuated targets exist for most utterances; missing ones are excluded from training targets only.

    ## Step 3 · Prediction problem (contract) - see PIPELINE.md
    """)
    return


if __name__ == "__main__":
    app.run()
