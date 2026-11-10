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
    sys.path.insert(0, str(ROOT / "ml_pipeline"))
    import guard
    from cleartalk_ml import corpora
    from cleartalk_ml import pipeline_log as log
    from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES, RESULTS
    from cleartalk_ml.text import normalize, script_of

    STATE = ROOT / "ml_pipeline"
    COLORS = {"kk": "#2a9d8f", "ru": "#3b6ea8", "en": "#d08c2b"}
    ROLE_COLORS = {"train": "#3b6ea8", "val": "#d08c2b", "test": "#5a9e5a"}
    return (COLORS, Counter, LANGUAGES, LANGUAGE_NAMES, RESULTS, ROLE_COLORS, ROOT, STATE, corpora, guard, json, log, mo,
            normalize, np, pd, plt, script_of)


@app.cell
def _(mo):
    mo.md("""
    # Steps 1-3 · Data inspection, EDA and the prediction problem (multilingual model)

    **Unit of data:** one recorded utterance with its transcript, in Kazakh, Russian or English.
    Later (step 5) several utterances in different languages are placed one after another to
    imitate a recording where the language changes. Nothing is modified in this notebook.
    """)
    return


@app.cell
def _(corpora):
    # Raw data, read-only: the table is rebuilt from the unpacked archives when it is missing.
    if not corpora.MANIFEST.is_file():
        corpora.main()
    utterances = corpora.load_manifest()
    utterances["words"] = utterances.text_raw.str.split().str.len()
    utterances["chars"] = utterances.text_raw.str.len()
    utterances.groupby(["corpus", "language", "split"]).agg(
        utterances=("utt_id", "size"), hours=("duration", lambda s: round(s.sum() / 3600, 2)),
        median_seconds=("duration", "median"), domains=("domain", "nunique"))
    return (utterances,)


@app.cell
def _(STATE, guard, utterances):
    # Step 1: the manifest table is profiled (text and audio themselves are too large for a table).
    profile = guard.profile(
        utterances[["corpus", "language", "split", "domain", "group", "duration", "gender", "words", "chars", "text_raw"]],
        target="text_raw", group_col="group")
    {"shape": profile["shape"], "duplicates": profile["duplicates"],
     "leakage_suspects": profile["leakage_suspects"], "traits": profile["traits"]}
    return (profile,)


@app.cell
def _(normalize, utterances):
    # Leakage checks that matter for speech recognition: the same sentence on both sides of a split.
    # FLEURS gives one id to a sentence, shared by every reader and by its translations.
    _fleurs = utterances[utterances.corpus == "fleurs"]
    _ids = _fleurs.groupby("split")["sentence_id"].apply(set)
    sentence_overlap = {f"{a}&{b}": len(_ids.get(a, set()) & _ids.get(b, set()))
                        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    _keyed = utterances.assign(key=utterances.text_raw.map(normalize))
    _texts = _keyed[_keyed.key.str.split().str.len() >= 4].groupby("split")["key"].apply(set)
    text_overlap = {f"{a}&{b}": len(_texts.get(a, set()) & _texts.get(b, set()))
                    for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    missing_audio = int((~utterances.exists.astype(str).str.lower().eq("true")).sum())
    empty_text = int((utterances.text_raw.str.strip() == "").sum())
    {"fleurs_sentence_ids_shared": sentence_overlap, "identical_sentences_shared": text_overlap,
     "missing_audio_files": missing_audio, "empty_transcripts": empty_text}
    return empty_text, missing_audio, sentence_overlap, text_overlap


@app.cell
def _(LANGUAGE_NAMES, ROLE_COLORS, log, np, plt, utterances):
    _table = utterances.groupby(["language", "corpus", "split"]).duration.sum().div(3600).unstack("split").fillna(0)
    _table = _table.reindex(columns=["train", "val", "test"], fill_value=0)
    _labels = [f"{LANGUAGE_NAMES[lang]}\n{corpus}" for lang, corpus in _table.index]
    _f, _axes = plt.subplots(1, 2, figsize=(12, 3.8))
    _x = np.arange(len(_table))
    for _i, _role in enumerate(_table.columns):
        _axes[0].bar(_x + (_i - 1) * 0.27, _table[_role], 0.27, label=_role, color=ROLE_COLORS[_role])
    _axes[0].set_xticks(_x, _labels, fontsize=8)
    _axes[0].set_ylabel("hours of speech")
    _axes[0].set_title("Audio per language, corpus and official part")
    _axes[0].legend()
    _counts = utterances.groupby(["language", "corpus"]).size()
    _axes[1].bar([f"{LANGUAGE_NAMES[lang]}\n{corpus}" for lang, corpus in _counts.index], _counts.to_numpy(), color="#777777")
    _axes[1].set_ylabel("utterances")
    _axes[1].set_title("Utterances per language and corpus")
    _axes[1].tick_params(axis="x", labelsize=8)
    _hours = utterances.groupby("language").duration.sum().div(3600)
    _train = utterances[utterances.split == "train"].groupby("language").duration.sum().div(3600)
    log.fig(1, "corpus_overview", _f,
              "Hours per language in all parts: " + ", ".join(f"{LANGUAGE_NAMES[k]} {v:.1f} h" for k, v in _hours.items())
              + "; training part: " + ", ".join(f"{LANGUAGE_NAMES[k]} {v:.1f} h" for k, v in _train.items())
              + ". English has only FLEURS, so the three languages are unequal and training must be balanced by "
              "sampling, not by taking everything.")
    return


@app.cell
def _(empty_text, log, missing_audio, plt, utterances):
    _checks = {
        "audio file missing": missing_audio,
        "empty transcript": empty_text,
        "shorter than 1 s": int((utterances.duration < 1).sum()),
        "longer than 30 s\n(Whisper window)": int((utterances.duration > 30).sum()),
        "no gender label": int((utterances.gender.fillna("") == "").sum()),
        "transcript without\npunctuation or capitals": int((~utterances.text_raw.str.contains(r"[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ.,?!]", regex=True)).sum()),
    }
    _f, _ax = plt.subplots(figsize=(8, 3.4))
    _ax.barh(list(_checks)[::-1], [100 * v / len(utterances) for v in _checks.values()][::-1], color="#9b4d4d")
    for _i, (_k, _v) in enumerate(list(_checks.items())[::-1]):
        _ax.text(100 * _v / len(utterances) + 0.5, _i, f"{_v:,}", va="center", fontsize=8)
    _ax.set_xlabel(f"% of {len(utterances):,} utterances")
    _ax.set_title("What is missing or unusual in the raw data")
    _plain = _checks["transcript without\npunctuation or capitals"]
    log.fig(1, "missing_and_unusual", _f,
              f"No audio file is missing ({missing_audio}) and {empty_text} transcripts are empty. "
              f"{_plain:,} utterances ({100 * _plain / len(utterances):.0f}%) have plain lower-case transcripts without "
              "punctuation (KSC2 and Golos), while FLEURS keeps capitals and punctuation: the corpora follow two "
              "writing styles, which decides what the model is taught to write. "
              f"{_checks['longer than 30 s' + chr(10) + '(Whisper window)']:,} utterances exceed Whisper's 30 s window and "
              f"{_checks['no gender label']:,} have no gender label (only FLEURS has one; no corpus has speaker ids).",
              )
    return


@app.cell
def _(COLORS, LANGUAGE_NAMES, log, np, plt, utterances):
    _f, _axes = plt.subplots(1, 2, figsize=(12, 3.6))
    _bins = np.linspace(0, 40, 81)
    for (_corpus, _lang), _group in utterances.groupby(["corpus", "language"]):
        _axes[0].hist(_group.duration.clip(upper=40), bins=_bins, histtype="step", linewidth=1.5, density=True,
                      color=COLORS[_lang], linestyle={"fleurs": "-", "ksc2": "--", "golos": ":"}[_corpus],
                      label=f"{LANGUAGE_NAMES[_lang]} · {_corpus}")
    _axes[0].axvline(30, color="black", linewidth=0.8)
    _axes[0].set_xlabel("utterance duration, seconds (30 s = Whisper window)")
    _axes[0].set_ylabel("density")
    _axes[0].set_title("Utterance duration by corpus")
    _axes[0].legend(fontsize=7)
    _med = utterances.groupby(["corpus", "language"]).duration.median()
    _axes[1].bar([f"{LANGUAGE_NAMES[lang]}\n{corpus}" for corpus, lang in _med.index], _med.to_numpy(),
                 color=[COLORS[lang] for _, lang in _med.index])
    _axes[1].set_ylabel("median duration, seconds")
    _axes[1].set_title("Median utterance duration")
    _axes[1].tick_params(axis="x", labelsize=8)
    log.fig(2, "duration_by_corpus", _f,
              "Median utterance duration: " + ", ".join(f"{corpus} {LANGUAGE_NAMES[lang]} {v:.1f} s" for (corpus, lang), v in _med.items())
              + ". FLEURS sentences are long (two or three fill a 30 s window), Golos phrases are short commands: a "
              "mixed-language recording built from FLEURS holds only 2-3 language changes, and short Golos phrases "
              "are the only short-utterance material.")
    return


@app.cell
def _(COLORS, LANGUAGE_NAMES, log, plt, utterances):
    _ok = utterances[(utterances.duration > 0.5) & (utterances.words > 0)]
    _rate = (_ok.words / _ok.duration).groupby([_ok.corpus, _ok.language]).median()
    _cps = (_ok.chars / _ok.duration).groupby([_ok.corpus, _ok.language]).median()
    _wordlen = (_ok.chars / _ok.words).groupby([_ok.corpus, _ok.language]).median()
    _f, _axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for _ax, _series, _title in zip(_axes, (_rate, _cps, _wordlen),
                                    ("words per second", "characters per second", "characters per word")):
        _ax.bar([f"{LANGUAGE_NAMES[lang]}\n{corpus}" for corpus, lang in _series.index], _series.to_numpy(),
                color=[COLORS[lang] for _, lang in _series.index])
        _ax.set_title(f"Median {_title}")
        _ax.tick_params(axis="x", labelsize=7)
    _by_lang = (_ok.chars / _ok.words).groupby(_ok.language).median()
    log.fig(2, "speaking_rate_and_word_length", _f,
              "Median characters per word: " + ", ".join(f"{LANGUAGE_NAMES[k]} {v:.1f}" for k, v in _by_lang.items())
              + ". Kazakh and Russian words are longer than English ones (endings are glued to the word), so one "
              "wrong ending costs a whole word: word error rate is harsher for them, which is why character error "
              "rate is reported next to it.")
    return


@app.cell
def _(Counter, LANGUAGES, LANGUAGE_NAMES, log, normalize, np, plt, script_of, utterances):
    # How much language mixing already exists inside the transcripts (borrowed words, names, digits).
    _share = {}
    for _lang in LANGUAGES:
        _sample = utterances[utterances.language == _lang]
        _sample = _sample.sample(min(8000, len(_sample)), random_state=0)
        _counter = Counter()
        for _text in _sample.text_raw:
            for _word in normalize(_text).split():
                _counter["digits" if _word.isdigit() else script_of(_word)] += 1
        _total = sum(_counter.values())
        _share[_lang] = {k: _counter[k] / _total for k in ("kk", "cyr", "lat", "digits", "other")}
    _f, _ax = plt.subplots(figsize=(8.5, 3.4))
    _bottom = np.zeros(len(LANGUAGES))
    _names = {"kk": "Cyrillic with Kazakh-only letters", "cyr": "other Cyrillic", "lat": "Latin", "digits": "digits", "other": "other"}
    for _kind, _color in zip(_names, ("#2a9d8f", "#3b6ea8", "#d08c2b", "#888888", "#cccccc")):
        _vals = np.array([_share[lang][_kind] for lang in LANGUAGES])
        _ax.barh([LANGUAGE_NAMES[lang] for lang in LANGUAGES], 100 * _vals, left=100 * _bottom, color=_color, label=_names[_kind])
        _bottom += _vals
    _ax.set_xlabel("% of words in the transcripts")
    _ax.set_title("Writing systems inside each language's transcripts")
    _ax.legend(fontsize=7, loc="lower right")
    log.fig(2, "scripts_inside_transcripts", _f,
              f"Latin-script words are {100 * _share['kk']['lat']:.2f}% of Kazakh and {100 * _share['ru']['lat']:.2f}% of Russian "
              f"transcript words, and digits {100 * _share['kk']['digits']:.2f}% / {100 * _share['ru']['digits']:.2f}% / "
              f"{100 * _share['en']['digits']:.2f}% (kk/ru/en). The corpora contain almost no natural language mixing, so "
              "mixed speech has to be constructed (step 5), and only "
              f"{100 * _share['kk']['kk']:.0f}% of Kazakh words carry a Kazakh-only letter: script alone cannot tell "
              "Kazakh from Russian.")
    return


@app.cell
def _(LANGUAGES, LANGUAGE_NAMES, log, normalize, np, plt, utterances):
    _oov, _types = {}, {}
    for _lang in LANGUAGES:
        _rows = utterances[utterances.language == _lang]
        _train = set(w for t in _rows[_rows.split == "train"].text_raw for w in normalize(t).split())
        _types[_lang] = len(_train)
        for _role in ("val", "test"):
            _words = [w for t in _rows[_rows.split == _role].text_raw for w in normalize(t).split()]
            _oov[(_lang, _role)] = float(np.mean([w not in _train for w in _words])) if _words and _train else float("nan")
    _f, _axes = plt.subplots(1, 2, figsize=(11, 3.4))
    _axes[0].bar([LANGUAGE_NAMES[lang] for lang in LANGUAGES], [_types[lang] for lang in LANGUAGES], color="#777777")
    _axes[0].set_title("Distinct words in the training transcripts")
    _x = np.arange(len(LANGUAGES))
    _axes[1].bar(_x - 0.2, [100 * _oov[(lang, "val")] for lang in LANGUAGES], 0.4, label="validation", color="#d08c2b")
    _axes[1].bar(_x + 0.2, [100 * _oov[(lang, "test")] for lang in LANGUAGES], 0.4, label="test", color="#5a9e5a")
    _axes[1].set_xticks(_x, [LANGUAGE_NAMES[lang] for lang in LANGUAGES])
    _axes[1].set_ylabel("% of running words never seen in training text")
    _axes[1].set_title("Words outside the training vocabulary")
    _axes[1].legend()
    log.fig(2, "vocabulary_and_unseen_words", _f,
              "Running words of the validation part that never occur in the training transcripts: "
              + ", ".join(f"{LANGUAGE_NAMES[lang]} {100 * _oov[(lang, 'val')]:.1f}%" for lang in LANGUAGES)
              + ". Many words must come from Whisper's pretrained knowledge, not from our training text - a reason to "
              "adapt Whisper gently (LoRA) instead of retraining it, and a warning that Kazakh, with its many word "
              "forms, depends most on how much text the model has already seen.")
    return


@app.cell
def _(COLORS, LANGUAGES, LANGUAGE_NAMES, RESULTS, log, np, pd, plt):
    # Baseline error profile on VALIDATION: ready models through the app's decoding path (stage 1).
    from cleartalk_ml.evaluate import report as _report

    ready = _report(["fleurs-val"]).rename(columns={"system": "model"}) if (RESULTS / "hyp").is_dir() else pd.DataFrame()
    if len(ready):
        _models = list(dict.fromkeys(ready.model))
        _f, _axes = plt.subplots(1, 3, figsize=(14, 3.6))
        _x = np.arange(len(_models))
        for _i, _lang in enumerate(LANGUAGES):
            _rows = ready[ready.language == _lang].set_index("model").reindex(_models)
            _axes[0].bar(_x + (_i - 1) * 0.27, _rows["WER forced %"], 0.27, color=COLORS[_lang], label=LANGUAGE_NAMES[_lang])
            _axes[1].bar(_x + (_i - 1) * 0.27, _rows["CER forced %"], 0.27, color=COLORS[_lang])
            _axes[2].bar(_x + (_i - 1) * 0.27, _rows["language found %"], 0.27, color=COLORS[_lang])
        for _ax, _title in zip(_axes, ("Word error rate, language given (%)", "Character error rate, language given (%)",
                                       "Language detected correctly among kk/ru/en (%)")):
            _ax.set_xticks(_x, _models, fontsize=8)
            _ax.set_title(_title, fontsize=10)
        _axes[0].legend(fontsize=8)
        _best = ready.loc[ready.groupby("language")["WER forced %"].idxmin()].set_index("language")
        log.fig(2, "ready_models_validation", _f,
                  "Ready models on the FLEURS validation part, the app's decoding path. Lowest word error rate per "
                  "language: " + ", ".join(f"{LANGUAGE_NAMES[lang]} {_best.loc[lang, 'WER forced %']:.1f}% ({_best.loc[lang, 'model']})"
                                           for lang in LANGUAGES if lang in _best.index)
                  + (". Whisper small - the only size this laptop can train - is "
                     + ", ".join(f"{LANGUAGE_NAMES[lang]} {wer:.1f}%" for lang, wer in
                                 zip(ready[ready.model == "small"].language, ready[ready.model == "small"]["WER forced %"]))
                     + ": the gap to the large models, largest for Kazakh, is what fine-tuning has to close."
                     if (ready.model == "small").any() else ". Whisper small is still being measured."),
                  )
    ready
    return (ready,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 3 · The prediction problem

    The contract is written in `ml_pipeline/PIPELINE.md` ("Problem contract"). In short: from audio
    alone, write the words that were spoken, each in its own language and script; one unit is a window
    of at most 30 seconds that holds either one utterance or several utterances in different languages;
    the measure is the pooled word error rate after removing case and punctuation.
    """)
    return


if __name__ == "__main__":
    app.run()
