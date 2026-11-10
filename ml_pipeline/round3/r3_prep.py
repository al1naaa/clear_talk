import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
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
    os.environ["CLEARTALK_DATA"] = "kazakh-broadcast"   # this round's training composition
    from cleartalk_ml import analysis, round3, splits, validation
    from cleartalk_ml import pipeline_log as log
    from cleartalk_ml.corpora import load_manifest
    from cleartalk_ml.paths import RESULTS

    log.use(round3.R3)
    SOURCES = ["crowdsourced", "parliament", "podcasts", "radio", "talkshow", "tv_news"]
    round1 = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    SMALL_OURS = f"runs/{round1['run']}/ct2-step-{round1['chosen_step']:05d}"
    return RESULTS, SMALL_OURS, SOURCES, analysis, json, load_manifest, log, mo, np, pd, plt, round3, splits, validation


@app.cell
def _(mo):
    mo.md("""
    # Round 3 · Steps 1-8: Kazakh broadcast speech for the small model

    New in this round: the broadcast sources of the KSC2 training part, streamed on 2026-10-14, and test C.
    Everything else (FLEURS, Golos, the crowdsourced KSC2 speech, the cleaning rules, the validation sets) is
    round 1's. No model is fitted in this notebook.
    """)
    return


@app.cell
def _(SMALL_OURS, SOURCES, analysis, load_manifest, log, np, plt, splits):
    # Steps 1-2 - what was added, and why: the round-1 small model on KSC2 validation, source by source.
    table = splits.cleaned(load_manifest())
    ksc2_train = table[(table.corpus == "ksc2") & (table.role == "train")]
    by_source = ksc2_train.groupby("domain").agg(utterances=("utt_id", "size"), hours=("duration", lambda s: s.sum() / 3600),
                                                 kept=("keep", "sum"))
    rate = ksc2_train.text_raw.str.len() / ksc2_train.duration.clip(lower=0.1)
    weakness = analysis.ksc2_by_source(SMALL_OURS)
    _f, _axes = plt.subplots(1, 3, figsize=(17, 4.2))
    _names = [s for s in SOURCES if s in by_source.index]
    _axes[0].bar(_names, [by_source.loc[s, "hours"] for s in _names], color=["#999999" if s == "crowdsourced" else "#2a9d8f" for s in _names])
    _axes[0].set_ylabel("hours of training speech on disk")
    _axes[0].set_title("KSC2 training part on disk, by source\n(grey: what rounds 1 and 2 had)", fontsize=10)
    _axes[0].tick_params(axis="x", labelsize=8)
    for _s in _names:
        _axes[1].hist(ksc2_train[ksc2_train.domain == _s].duration.clip(upper=30), bins=np.arange(0, 31, 1), histtype="step", label=_s)
    _axes[1].set_xlabel("utterance length, s (cut at 30)")
    _axes[1].set_ylabel("utterances")
    _axes[1].set_title("Utterance lengths by source", fontsize=10)
    _axes[1].legend(fontsize=7)
    _axes[2].bar(_names, [100 * weakness[s]["wer"] for s in _names if s in weakness], color="#e9b384")
    _axes[2].set_ylabel("WER %, language given")
    _axes[2].set_title("Why: the round-1 small model on KSC2 VALIDATION, by source", fontsize=10)
    _axes[2].tick_params(axis="x", labelsize=8)
    _f.tight_layout()
    log.fig(2, "kazakh_broadcast_data", _f,
            "The KSC2 training speech on disk after the stream of 2026-10-14, by source: "
            + ", ".join(f"{s} {by_source.loc[s, 'utterances']:,} utterances ({by_source.loc[s, 'hours']:.1f} h)" for s in _names)
            + ". Median length by source: " + ", ".join(f"{s} {ksc2_train[ksc2_train.domain == s].duration.median():.1f} s" for s in _names)
            + "; characters per second by source: " + ", ".join(f"{s} {rate[ksc2_train.domain == s].median():.1f}" for s in _names)
            + ". The reason for this round, measured on validation in round 1: the small model's WER on KSC2 by source is "
            + ", ".join(f"{s} {100 * weakness[s]['wer']:.0f}%" for s in _names if s in weakness)
            + " - read speech is its best case, and the broadcast sources it never heard are its worst.")
    {s: {"utterances": int(by_source.loc[s, "utterances"]), "hours": round(float(by_source.loc[s, "hours"]), 2), "kept": int(by_source.loc[s, "kept"])} for s in _names}
    return ksc2_train, table


@app.cell
def _(SOURCES, ksc2_train, log, plt):
    # Step 4 - the cleaning rules of round 1, unchanged, applied to the new speech.
    _reasons = ksc2_train[~ksc2_train.keep].groupby(["domain", "reason"]).size().unstack(fill_value=0)
    _names = [s for s in SOURCES if s in ksc2_train.domain.unique()]
    _f, _ax = plt.subplots(figsize=(11, 3.8))
    _bottom = [0] * len(_names)
    for _reason in _reasons.columns:
        _values = [int(_reasons.loc[s, _reason]) if s in _reasons.index else 0 for s in _names]
        _ax.bar(_names, _values, bottom=_bottom, label=_reason)
        _bottom = [a + b for a, b in zip(_bottom, _values)]
    _ax.set_ylabel("utterances removed")
    _ax.set_title("Cleaning rules of round 1 on the KSC2 training speech, by source")
    _ax.legend(fontsize=8)
    _f.tight_layout()
    _removed = ksc2_train[~ksc2_train.keep]
    log.fig(4, "cleaning_kazakh_broadcast", _f,
            f"The cleaning rules of round 1, unchanged (length 1-30 s, 1-28 characters per second, empty transcript, a sentence that is "
            f"held out for validation or test): of {len(ksc2_train):,} KSC2 training utterances {len(_removed):,} are removed - "
            + ", ".join(f"{s} {int((_removed.domain == s).sum()):,} of {int((ksc2_train.domain == s).sum()):,}" for s in _names)
            + ". By reason: " + ", ".join(f"{reason} {count:,}" for reason, count in _removed.reason.value_counts().items()) + ".")
    _removed.groupby(["domain", "reason"]).size()
    return


@app.cell
def _(SOURCES, log, np, plt, round3, splits):
    # Steps 5-6 - the round-3 epoch and test C.
    from cleartalk_ml.mixed import load_plan

    epoch = splits.training_examples()
    singles = splits.role_table("train", "single")
    recipes = load_plan(splits.composition()["plan"])[: splits.composition()["mixed"]]
    pool = splits.role_table("train", "mix")
    domain_of = dict(zip(pool.utt_id, pool.domain))
    parts = [part for recipe in recipes for part in recipe["parts"]]
    kazakh_parts = [part for part in parts if part["utt_id"].startswith("ksc2-")]
    broadcast_parts = [part for part in kazakh_parts if domain_of.get(part["utt_id"]) not in (None, "crowdsourced")]
    units = round3.test_c_units()
    fresh = round3.check_fresh(units)
    _f, _axes = plt.subplots(1, 2, figsize=(14, 4))
    _groups = singles.assign(group=[f"{c}-{d}" if c == "ksc2" else c for c, d in zip(singles.corpus, singles.domain)]).group.value_counts().sort_index()
    _axes[0].bar(_groups.index, _groups.values, color=["#2a9d8f" if g.startswith("ksc2-") and not g.endswith("crowdsourced") else "#999999" for g in _groups.index])
    _axes[0].set_ylabel("single utterances per epoch")
    _axes[0].set_title(f"Round-3 epoch: {len(epoch):,} examples ({len(recipes):,} mixed recordings)", fontsize=10)
    _axes[0].tick_params(axis="x", labelsize=7, rotation=30)
    _sets = units.assign(group=[f"{s} · {d}" if d else s for s, d in zip(units.set, units.domain)]).group.value_counts().sort_index()
    _axes[1].barh(_sets.index, _sets.values, color="#de693c")
    _axes[1].set_xlabel("units")
    _axes[1].set_title(f"Test C: {len(units)} units, frozen before training", fontsize=10)
    _f.tight_layout()
    log.fig(6, "epoch_and_test_c", _f,
            f"The round-3 epoch: {len(epoch):,} examples - {len(singles):,} single utterances (" + ", ".join(f"{g} {n:,}" for g, n in _groups.items())
            + f") and {len(recipes):,} mixed recordings built by new seeded recipes from a pool that now also holds the kept broadcast speech: "
            f"{len(kazakh_parts):,} of their {len(parts):,} parts are KSC2, {len(broadcast_parts):,} of those broadcast. Test C: {len(units)} units - "
            + ", ".join(f"{g} {n}" for g, n in _sets.items()) + f"; shared with tests A and B, validation or training: "
            + ", ".join(f"{k} {v}" for k, v in fresh.items()) + ".")
    {"epoch": len(epoch), "singles": _groups.to_dict(), "mixed": len(recipes), "kazakh_parts": len(kazakh_parts), "broadcast_parts": len(broadcast_parts), "fresh": fresh}
    return


if __name__ == "__main__":
    app.run()
