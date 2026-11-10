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
    from cleartalk_ml import corpora, mixed, splits
    from cleartalk_ml import pipeline_log as log
    from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES

    COLORS = {"kk": "#2a9d8f", "ru": "#3b6ea8", "en": "#d08c2b"}
    ROLE_COLORS = {"train": "#3b6ea8", "val": "#d08c2b", "test": "#5a9e5a"}
    return COLORS, Counter, LANGUAGES, LANGUAGE_NAMES, ROLE_COLORS, ROOT, corpora, json, log, mixed, mo, np, pd, plt, splits


@app.cell
def _(mo):
    mo.md("""
    # Steps 4-8 · From raw utterances to training examples

    Cleaning, building mixed-language recordings, fixing the roles (train / validation / test),
    and turning audio and text into what Whisper reads and writes. No model is fitted here.
    """)
    return


@app.cell
def _(corpora, splits):
    # Step 4 - cleaning. Every rule looks only at the utterance itself (or at the fact that its
    # sentence is held out); validation and test utterances are never filtered by quality.
    table = splits.cleaned(corpora.load_manifest())
    funnel = table.groupby(["corpus", "language", "role"]).agg(before=("utt_id", "size"), after=("keep", "sum"))
    reasons = table[~table.keep].groupby(["role", "reason"]).size().rename("utterances")
    (funnel, reasons)
    return funnel, reasons, table


@app.cell
def _(LANGUAGE_NAMES, log, plt, reasons, table):
    _train = table[table.role == "train"]
    _by_reason = _train[~_train.keep].groupby("reason").size().sort_values()
    _f, _axes = plt.subplots(1, 2, figsize=(13, 3.6))
    _axes[0].barh(_by_reason.index, _by_reason.to_numpy(), color="#9b4d4d")
    for _i, _v in enumerate(_by_reason.to_numpy()):
        _axes[0].text(_v, _i, f" {_v:,}", va="center", fontsize=8)
    _axes[0].set_title(f"Training utterances removed, by rule ({len(_train):,} before)")
    _axes[0].set_xlabel("utterances")
    _counts = _train.groupby(["corpus", "language"]).agg(before=("utt_id", "size"), after=("keep", "sum"))
    _labels = [f"{LANGUAGE_NAMES[lang]}\n{corpus}" for corpus, lang in _counts.index]
    _axes[1].bar(_labels, _counts.before, color="#cccccc", label="before")
    _axes[1].bar(_labels, _counts.after, color="#3b6ea8", width=0.5, label="after cleaning")
    _axes[1].set_title("Training utterances per corpus")
    _axes[1].tick_params(axis="x", labelsize=8)
    _axes[1].legend()
    _held = table[(table.role != "train") & ~table.keep].groupby("reason").size()
    _plain = int(table.corpus.isin(["golos", "ksc2"]).sum())
    log.fig(4, "cleaning_funnel", _f,
            f"Training utterances {len(_train):,} → {int(_train.keep.sum()):,} ({100 * _train.keep.mean():.1f}% kept). Removed: "
            + "; ".join(f"{k} {v:,}" for k, v in _by_reason.sort_values(ascending=False).items())
            + ". Validation and test utterances are removed only when they cannot be scored ("
            + (", ".join(f"{k} {v:,}" for k, v in _held.items()) or "none") + f"). The {_plain:,} plain lower-case "
            "transcripts (Golos, KSC2) get sentence form as training targets - a capital first letter and a full "
            "stop - so the model is taught one writing style; scoring ignores case and punctuation anyway.")
    reasons
    return


@app.cell
def _(mixed, splits):
    # Step 5 - data engineering: frozen recipes of mixed-language recordings, one set per role.
    for _split in ("train", "val", "test"):
        if not mixed.plan_path(_split).is_file():
            mixed.save_plan(_split, mixed.make_plan(splits.role_table(_split, "mix"), mixed.SIZES[_split],
                                                    mixed.SEEDS[_split], f"mixed-{_split}"))
    plans = {split: mixed.load_plan(split) for split in ("train", "val", "test")}
    # Data-quality checks on every recipe.
    for _split, _recipes in plans.items():
        assert len({r["mix_id"] for r in _recipes}) == len(_recipes) == mixed.SIZES[_split]
        for _r in _recipes:
            _langs = [p["language"] for p in _r["parts"]]
            assert 2 <= len(_langs) <= 4 and all(a != b for a, b in zip(_langs, _langs[1:]))
            assert _r["duration"] <= mixed.MAX_SECONDS + 1e-6
            assert all(b["start"] >= a["end"] for a, b in zip(_r["parts"], _r["parts"][1:]))
    mixed.render_split("val")  # validation recordings as WAV files, for the app's decoding path
    {split: len(recipes) for split, recipes in plans.items()}
    return (plans,)


@app.cell
def _(COLORS, Counter, LANGUAGE_NAMES, LANGUAGES, ROLE_COLORS, log, np, plans, plt):
    _f, _axes = plt.subplots(1, 4, figsize=(16, 3.5))
    _roles = list(plans)
    for _i, _role in enumerate(_roles):
        _parts = Counter(len(r["parts"]) for r in plans[_role])
        _axes[0].bar(np.array([2, 3, 4]) + (_i - 1) * 0.27, [100 * _parts[k] / len(plans[_role]) for k in (2, 3, 4)], 0.27,
                     color=ROLE_COLORS[_role], label=_role)
        _axes[1].hist([r["duration"] for r in plans[_role]], bins=np.linspace(0, 30, 31), histtype="step", density=True,
                      color=ROLE_COLORS[_role], linewidth=1.5)
    _axes[0].set_xticks([2, 3, 4])
    _axes[0].set_xlabel("utterances in one recording")
    _axes[0].set_ylabel("% of recordings")
    _axes[0].set_title("Language changes per recording")
    _axes[0].legend()
    _axes[1].set_xlabel("recording duration, s")
    _axes[1].set_title("Duration of mixed recordings")
    _share = {role: Counter(p["language"] for r in plans[role] for p in r["parts"]) for role in _roles}
    for _i, _lang in enumerate(LANGUAGES):
        _axes[2].bar(np.arange(len(_roles)) + (_i - 1) * 0.27,
                     [100 * _share[role][_lang] / sum(_share[role].values()) for role in _roles], 0.27,
                     color=COLORS[_lang], label=LANGUAGE_NAMES[_lang])
    _axes[2].set_xticks(np.arange(len(_roles)), _roles)
    _axes[2].set_ylabel("% of parts")
    _axes[2].set_title("Language of the parts")
    _axes[2].legend(fontsize=7)
    _source = {role: Counter(p["utt_id"].split("-")[0] for r in plans[role] for p in r["parts"]) for role in _roles}
    _corpora = sorted({c for counter in _source.values() for c in counter})
    _bottom = np.zeros(len(_roles))
    for _corpus, _color in zip(_corpora, ("#777777", "#b0b0b0", "#d9d9d9")):
        _vals = np.array([100 * _source[role][_corpus] / sum(_source[role].values()) for role in _roles])
        _axes[3].bar(_roles, _vals, bottom=_bottom, color=_color, label=_corpus)
        _bottom += _vals
    _axes[3].set_ylabel("% of parts")
    _axes[3].set_title("Corpus of the parts")
    _axes[3].legend(fontsize=7)
    _train = plans["train"]
    _mean_parts = {role: np.mean([len(r["parts"]) for r in plans[role]]) for role in _roles}
    log.fig(5, "mixed_recordings", _f,
            f"Mixed-language recordings: {len(plans['train']):,} for training, {len(plans['val'])} for validation, "
            f"{len(plans['test'])} for the test, each built only from utterances of its own role. A recording holds "
            + ", ".join(f"{_mean_parts[role]:.2f}" for role in _roles) + " utterances on average (train / val / test) and lasts "
            f"{np.median([r['duration'] for r in _train]):.1f} s (median, training). Every neighbouring pair is in different "
            "languages, so each recording has at least one language change; because every corpus is monolingual, a "
            "language change is always also a change of speaker and microphone - real switching inside one "
            "person's sentence is not represented.")
    return


@app.cell
def _(LANGUAGE_NAMES, ROLE_COLORS, log, np, pd, plt, splits):
    # Step 6 - roles. Nothing is fitted before this point; the test units are frozen now.
    overlap = splits.check_disjoint()
    units = {role: splits.eval_units(role) for role in ("val", "test")}
    frozen = splits.freeze_test(units["test"])
    train_examples = splits.training_examples()
    _rows = []
    for _example in train_examples:
        _rows.append({"role": "train", "set": "mixed" if _example["kind"] == "mixed" else f"{_example['corpus']}-{_example['language']}",
                      "duration": _example["duration"]})
    for _role, _frame in units.items():
        _rows += [{"role": _role, "set": s, "duration": d} for s, d in zip(_frame.set, _frame.duration)]
    sizes = pd.DataFrame(_rows).groupby(["set", "role"]).agg(units=("duration", "size"), hours=("duration", lambda s: s.sum() / 3600)).unstack("role")
    _sets = list(sizes.index)
    _f, _axes = plt.subplots(1, 2, figsize=(14, 3.8))
    _x = np.arange(len(_sets))
    for _i, _role in enumerate(("train", "val", "test")):
        _axes[0].bar(_x + (_i - 1) * 0.27, sizes[("units", _role)].fillna(0), 0.27, color=ROLE_COLORS[_role], label=_role)
        _axes[1].bar(_x + (_i - 1) * 0.27, sizes[("hours", _role)].fillna(0), 0.27, color=ROLE_COLORS[_role])
    for _ax, _title in zip(_axes, ("Units per set and role (log scale)", "Hours per set and role")):
        _ax.set_xticks(_x, _sets, rotation=20, fontsize=8)
        _ax.set_title(_title)
    _axes[0].set_yscale("log")
    _axes[0].legend()
    _n = {role: int(sizes[("units", role)].sum()) for role in ("train", "val", "test")}
    log.fig(6, "roles", _f,
            f"{_n['train']:,} training examples per epoch, {_n['val']:,} validation units and {_n['test']:,} test units "
            f"(a unit is one utterance or one mixed recording). Shared between roles: {sum(overlap.values())} utterances or "
            "FLEURS sentences (checked for single utterances and for the parts of mixed recordings). Roles follow the "
            "corpora's official parts; no corpus publishes speaker ids, so speaker overlap between roles cannot be "
            f"ruled out. The {frozen['n_test']:,} test units are frozen by fingerprint and stay unopened until step 14.")
    (overlap, frozen, sizes)
    return train_examples, units


@app.cell
def _(LANGUAGE_NAMES, log, np, plt, train_examples):
    # Step 7 - features: Whisper's fixed log-mel spectrogram. Nothing is estimated from data.
    from cleartalk_ml.dataset import example_audio, features

    _picked = [next(e for e in train_examples if e["kind"] == "single" and e["language"] == lang and e["corpus"] == "fleurs")
               for lang in ("kk", "ru", "en")]
    _picked.append(next(e for e in train_examples if e["kind"] == "mixed" and len(e["parts"]) == 3))
    _f, _axes = plt.subplots(len(_picked), 1, figsize=(13, 2.1 * len(_picked)), sharex=True)
    for _ax, _example in zip(_axes, _picked):
        _mel = features(example_audio(_example))
        _ax.imshow(_mel, origin="lower", aspect="auto", extent=(0, 30, 0, 80), cmap="magma")
        _ax.set_ylabel("mel band")
        if _example["kind"] == "mixed":
            for _part in _example["parts"]:
                _ax.axvline(_part["start"], color="white", linewidth=1)
                _ax.text(_part["start"] + 0.2, 70, LANGUAGE_NAMES[_part["language"]], color="white", fontsize=9)
            _ax.set_title("Mixed recording: one 30 s window, language changes at the white lines", fontsize=9)
        else:
            _ax.set_title(f"{LANGUAGE_NAMES[_example['language']]} utterance, {_example['duration']:.1f} s (the rest of the window is padding)", fontsize=9)
    _axes[-1].set_xlabel("seconds")
    _shape = features(np.zeros(16000, dtype=np.float32)).shape
    log.fig(7, "logmel_examples", _f,
            f"The model's input is Whisper's fixed log-mel spectrogram: {_shape[0]} bands × {_shape[1]} frames for a "
            "30-second window, computed by a fixed formula with no statistics estimated from our data - so features "
            "cannot leak anything from validation or test. A mixed recording fills most of the window and the language "
            "changes at visible pauses; a single utterance leaves most of the window as padding.")
    return


@app.cell
def _(Counter, log, np, plt, train_examples):
    # Step 8 - preprocessing: transcripts as Whisper tokens (targets), with or without timestamps.
    from cleartalk_ml.dataset import MAX_LABEL_TOKENS, prompt_language, target_tokens, uses_timestamps

    _lengths = {"single": [], "mixed": []}
    _stamped = 0
    _prompts = Counter()
    for _example in train_examples:
        _stamps = uses_timestamps(_example["id"])
        _stamped += _stamps
        _lengths[_example["kind"]].append(len(target_tokens(_example, _stamps)))
        if _example["kind"] == "mixed":
            _prompts["first language" if prompt_language(_example) == _example["parts"][0]["language"] else "another language"] += 1
    token_lengths = {kind: np.array(values) for kind, values in _lengths.items()}
    _f, _axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for _kind, _color in (("single", "#3b6ea8"), ("mixed", "#de693c")):
        _axes[0].hist(token_lengths[_kind], bins=np.linspace(0, MAX_LABEL_TOKENS, 56), histtype="step", linewidth=1.6,
                      color=_color, label=f"{_kind} ({len(token_lengths[_kind]):,})")
    _axes[0].axvline(MAX_LABEL_TOKENS, color="black", linewidth=0.8)
    _axes[0].set_xlabel("target length in tokens (black line = limit)")
    _axes[0].set_title("Length of training targets")
    _axes[0].legend()
    _axes[1].bar(list(_prompts), [100 * v / sum(_prompts.values()) for v in _prompts.values()], color="#777777")
    _axes[1].set_ylabel("% of mixed recordings")
    _axes[1].set_title("Language token given for a mixed recording")
    _all = np.concatenate(list(token_lengths.values()))
    _cut = int((_all >= MAX_LABEL_TOKENS).sum())
    log.fig(8, "target_tokens", _f,
            f"Targets are Whisper BPE tokens: median {int(np.median(token_lengths['single']))} tokens for a single utterance and "
            f"{int(np.median(token_lengths['mixed']))} for a mixed recording, longest {int(_all.max())}; {_cut} of {len(_all):,} "
            f"reach the {MAX_LABEL_TOKENS}-token limit and are cut. {100 * _stamped / len(_all):.0f}% of examples keep "
            "timestamps (one timed segment per utterance), so the app's timed segments survive fine-tuning. A mixed "
            f"recording gets one language token: its first language in {100 * _prompts['first language'] / max(sum(_prompts.values()), 1):.0f}% "
            "of cases, otherwise another language it contains - the model must write each part in its own language "
            "whatever the token says. The tokenizer and the feature settings are saved with the model, so training "
            "and the app cannot drift apart.")
    {kind: (int(np.median(v)), int(v.max())) for kind, v in token_lengths.items()}
    return


if __name__ == "__main__":
    app.run()
