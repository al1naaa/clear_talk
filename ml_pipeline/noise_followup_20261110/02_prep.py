"""Prepare restored train/dev data and deterministic augmentation recipes.

No fitting, checkpoint selection, or final-test reuse. Open with marimo.
"""
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import importlib.util
    import hashlib
    import json
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    sys.path.insert(0, str(root))
    from cleartalk_ml.corpora import fleurs_table
    from cleartalk_ml.text import normalize
    spec = importlib.util.spec_from_file_location("noise_prep_guard", here / "guard.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return fleurs_table, guard, hashlib, here, json, normalize, np, pd, plt, root


@app.cell
def _(fleurs_table, guard, here, json, normalize, pd, plt, root):
    if not (here / "restore_receipt.json").is_file():
        raise RuntimeError("Finish scripts/restore_noise_data.py before preparation")
    table = fleurs_table()
    if set(table.split) != {"train", "val"}:
        raise RuntimeError("This workbench accepts restored train/dev only; test stays separate")
    table["normalized"] = table.text_raw.map(normalize)
    table["excluded_reason"] = ""
    validation = table[table.split == "val"]
    demo = json.loads((root / "data/ml/plans/demo-utterances.json").read_text())
    demo_ids = {row["utt_id"] for row in demo["utterances"]}
    rules = [
        ("missing_audio", ~table.exists),
        ("empty_transcript", table.normalized.eq("")),
        ("duration_outside_1_30_seconds", ~table.duration.between(1, 30)),
        ("demo", table.utt_id.isin(demo_ids)),
        ("validation_sentence", table.group.isin(set(validation.group))),
        ("validation_text", table.normalized.isin(set(validation.normalized))),
        ("duplicate_training_utterance", table.duplicated("utt_id")),
    ]
    removed = {}
    for reason, mask in rules:
        hit = table.split.eq("train") & table.excluded_reason.eq("") & mask
        removed[reason] = int(hit.sum())
        table.loc[hit, "excluded_reason"] = reason
    kept = table[table.excluded_reason.eq("")].copy()
    if not kept.exists.all() or kept.normalized.eq("").any():
        raise RuntimeError("Invalid validation input: report it, never silently filter it")
    train = kept[kept.split == "train"]
    val = kept[kept.split == "val"]
    assert not set(train.utt_id) & set(val.utt_id)
    assert not set(train.group) & set(val.group)
    assert not set(train.normalized) & set(val.normalized)
    figure, ax = plt.subplots(figsize=(9, 4))
    counts = [int(table.split.eq("train").sum()), len(train), len(val)]
    ax.bar(["Train before", "Train retained", "Validation unchanged"], counts, color=["#94a3b8", "#0f766e", "#2563eb"])
    for index, value in enumerate(counts):
        ax.text(index, value, str(value), ha="center", va="bottom")
    ax.set_ylabel("Utterances")
    ax.set_title("Training cleaning; validation is preserved")
    guard.fig(4, "cleaning_funnel", figure,
              f"Training {counts[0]} to {counts[1]}; validation {counts[2]} unchanged. Exclusions: {removed}.", here)
    output = root / "data/ml/noise_followup_20261110"
    output.mkdir(exist_ok=True)
    columns = ["utt_id", "language", "split", "group", "path", "text_raw", "duration", "excluded_reason"]
    table[columns].to_csv(output / "cleaning_audit.csv", index=False)
    kept[columns].to_csv(output / "development_manifest.csv", index=False)
    summary = {"train_before": counts[0], "train_after": counts[1], "validation": counts[2],
               "removed_by_rule": removed, "train_val_shared_utterances": 0,
               "train_val_shared_sentence_groups": 0, "train_val_shared_normalized_texts": 0,
               "speaker_disjointness": "unknown; corpus has no reader IDs",
               "final_test": "not allocated; do not reuse previous test rounds"}
    (here / "cleaning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary
    return output, train, val


@app.cell
def _(guard, hashlib, here, json, np, output, plt, train, val):
    recipes = []
    for role, frame in (("train", train), ("val", val)):
        for row in frame.sort_values("utt_id").itertuples():
            seed = int.from_bytes(hashlib.sha256(f"noise-v1:{role}:{row.utt_id}".encode()).digest()[:4], "big")
            rng = np.random.default_rng(seed)
            kinds = ["clean", str(rng.choice(["white", "pink", "echo"]))] if role == "train" else ["clean", "white", "pink", "echo"]
            for kind in kinds:
                recipes.append({"unit_id": f"{role}:{row.utt_id}:{kind}", "source_id": row.utt_id,
                                "source_group": row.group, "role": role, "language": row.language,
                                "path": row.path, "reference": row.text_raw, "condition": kind,
                                "seed": seed, "snr_db": (float(rng.uniform(5, 20)) if role == "train" else 10.0) if kind in {"white", "pink"} else None,
                                "echo_delay_ms": 70 if kind == "echo" else None,
                                "echo_gain": 0.2 if kind == "echo" else None})
    path = output / "augmentation_recipes.jsonl"
    with path.open("w") as handle:
        for recipe in recipes:
            handle.write(json.dumps(recipe, ensure_ascii=False) + "\n")
    aug_counts = {kind: sum(r["role"] == "train" and r["condition"] == kind for r in recipes)
              for kind in ("clean", "white", "pink", "echo")}
    assert aug_counts["clean"] == len(train)
    assert sum(aug_counts.values()) == 2 * len(train)
    aug_figure, aug_ax = plt.subplots(figsize=(8, 4))
    aug_ax.bar(aug_counts.keys(), aug_counts.values(), color=["#0f766e", "#38bdf8", "#818cf8", "#f59e0b"])
    aug_ax.set_ylabel("Training recipe views")
    aug_ax.set_title("Exactly half of training views stay clean")
    guard.fig(5, "augmentation_plan", aug_figure,
              f"Counts: {aug_counts}. Every source has one clean and one augmented view. "
              "Synthetic noise and a mild echo are a preparation pilot; real noise and two-voice separation remain separate checks.", here)
    (here / "recipe_summary.json").write_text(json.dumps({
        "training_views": aug_counts, "validation_views": 4 * len(val),
        "recipes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rendered": False, "fitted": False,
        "step_6_blocker": "A new independent test source is needed. Current development recordings cannot certify real-world overlap quality."
    }, indent=2) + "\n")
    return


if __name__ == "__main__":
    app.run()
