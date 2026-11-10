"""Read-only inventory for the next noise experiment. No fitting or test scoring.

Open with: .venv-train/bin/marimo edit ml_pipeline/noise_followup_20261110/01_eda.py
"""
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import importlib.util
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    spec = importlib.util.spec_from_file_location("noise_inventory_guard", here / "guard.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return Path, guard, here, json, pd, plt, root


@app.cell
def _(guard, here, json, pd, root):
    rows = []
    for role in ("train", "val"):
        with (root / "data/ml/plans" / f"mixed-{role}.jsonl").open() as handle:
            for line in handle:
                recipe = json.loads(line)
                for part in recipe["parts"]:
                    rows.append({
                        "role": role, "mixture_id": recipe["mix_id"],
                        "utterance_id": part["utt_id"], "language": part["language"],
                        "audio_path": part["path"], "audio_exists": (root / part["path"]).is_file(),
                        "duration_s": part["end"] - part["start"],
                        "transcript": part["text"], "speaker_id": None,
                    })
    manifest = pd.DataFrame(rows)
    guard.profile(manifest, target="transcript", group_col="utterance_id", state_dir=here)
    profile_report = json.loads((here / "data_profile.json").read_text())
    profile_report
    return manifest, profile_report


@app.cell
def _(guard, here, json, manifest, plt, profile_report, root):
    # Classification plots describe corpus composition, not the ASR prediction target.
    assert profile_report["shape"]["n_rows"] == len(manifest)
    guard.eda_figures(manifest[["language", "duration_s", "speaker_id"]], target="language", state_dir=here)
    unique = manifest.drop_duplicates(["role", "utterance_id"])
    by_role = unique.groupby("role").audio_exists.agg(["count", "sum"])
    figure, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_role.index, by_role["count"], label="Required audio", color="#cbd5e1")
    ax.bar(by_role.index, by_role["sum"], label="Available audio", color="#14b8a6")
    for pos, count in enumerate(by_role["count"]):
        ax.text(pos, count, str(count), ha="center", va="bottom")
    ax.set_ylabel("Unique utterances referenced by saved recipes")
    ax.set_title("Training readiness on this computer")
    ax.legend()
    guard.fig(2, "audio_availability", figure,
              "Saved train/validation recipes reference audio that must be restored before training. "
              "Manifest completeness does not prove that the recordings are available.", here)
    normalize = lambda s: " ".join(s.lower().replace("ё", "е").split())
    sets = {role: set(unique.loc[unique.role == role, "utterance_id"]) for role in ("train", "val")}
    texts = {role: set(unique.loc[unique.role == role, "transcript"].map(normalize)) for role in ("train", "val")}
    inventory = {
        "date": "2026-11-10", "scope": "Saved train and validation recipes only; no test opened",
        "recipe_occurrences": len(manifest), "unique_by_role": by_role.to_dict("index"),
        "shared_utterance_ids_train_val": len(sets["train"] & sets["val"]),
        "shared_normalized_texts_train_val": len(texts["train"] & texts["val"]),
        "speaker_identity_available": False,
        "original_multilingual_manifest_available": (root / "data/ml/utterances.csv.gz").is_file(),
        "trainable_base_weights_available": (root / "models/multilingual/kazakh-whisper-large-v3-turbo-hf/model.safetensors").is_file(),
        "note": "Repeated occurrences are recipe reuse, not extra independent recordings. No files changed in source data.",
    }
    (here / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    inventory
    return


if __name__ == "__main__":
    app.run()
