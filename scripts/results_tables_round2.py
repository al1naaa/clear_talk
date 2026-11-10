"""Print the result tables of the large multilingual model (round 2) as Markdown, from the saved result files.

    python scripts/results_tables_round2.py            # English
    python scripts/results_tables_round2.py --ru       # Russian

Validation numbers come from the cached transcripts (data/ml/results/hyp), test numbers from the one-time
final test on test B (data/ml/results/test-final-B.json). Nothing is transcribed here.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from results_tables import NAMES, count, pct, points  # noqa: E402

SETS = ["mixed", "fleurs-kk", "ksc2-kk", "fleurs-ru", "golos-ru", "fleurs-en"]
HEAD = {
    "en": ("Set", "Kazakh turbo (starting model)", "**ClearTalk large (LoRA)**", "Change (95% bootstrap)", "Whisper large-v3-turbo",
           "ClearTalk small (round 1)", "App pipeline: language per turn / mixed setting"),
    "ru": ("Набор", "Казахская turbo (исходная)", "**ClearTalk большая (LoRA)**", "Изменение (95% интервал)", "Whisper large-v3-turbo",
           "ClearTalk малая (первый раунд)", "Конвейер приложения: язык по репликам / «языки меняются в реплике»"),
}


def validation_rows():
    from cleartalk_ml import validation
    from cleartalk_ml.evaluate import bootstrap_difference, details, units_of
    from cleartalk_ml.paths import RESULTS

    selection = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
    round1 = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    systems = {"start": "kazakh", "ours": f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}", "turbo": "turbo",
               "small_ours": f"runs/{round1['run']}/ct2-step-{round1['chosen_step']:05d}"}
    single = {name: validation.single_scores(system) for name, system in systems.items()}
    mixed = {name: validation.mixed_scores(system) for name, system in systems.items()}
    pipelines = [validation.mixed_scores(name) for name in ("pipeline-auto-r2", "pipeline-mixed-r2")]
    rows = {}
    for set_name in SETS:
        if set_name == "mixed":
            get = lambda name: mixed[name]["wer"] if mixed.get(name) else None  # noqa: E731
            compare = bootstrap_difference(mixed["start"]["details"], mixed["ours"]["details"])
            words, units = mixed["ours"]["reference_words"], mixed["ours"]["units"]
        else:
            get = lambda name: single[name][set_name]["forced"]["wer"] if set_name in single.get(name, {}) else None  # noqa: E731
            eval_set = {"fleurs": "fleurs-val", "ksc2": "ksc2-val", "golos": "golos-val"}[set_name.split("-")[0]]
            group = units_of(eval_set, language=set_name.split("-")[1])
            a, _ = validation.hypotheses(eval_set, systems["start"])
            b, _ = validation.hypotheses(eval_set, systems["ours"])
            compare = bootstrap_difference(details(group, a, "forced"), details(group, b, "forced"))
            words, units = single["ours"][set_name]["forced"]["reference_words"], single["ours"][set_name]["units"]
        rows[set_name] = {"start": get("start"), "ours": get("ours"), "compare": compare, "turbo": get("turbo"), "small_ours": get("small_ours"),
                          "pipelines": [block["wer"] if block else None for block in pipelines] if set_name == "mixed" else None,
                          "units": units, "words": words}
    return rows


def test_rows():
    from cleartalk_ml.paths import RESULTS

    path = RESULTS / "test-final-B.json"
    if not path.is_file():
        return None
    final = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for set_name in SETS:
        def get(system, set_name=set_name):
            for item in final["scores"]:
                if item["system"] == system and item["set"] == set_name:
                    return item["wer_auto"] if set_name == "mixed" else item["wer_forced"]
            return None

        ours = next(item for item in final["scores"] if item["system"] == "cleartalk-large" and item["set"] == set_name)
        rows[set_name] = {"start": get("kazakh"), "ours": get("cleartalk-large"), "compare": final["cleartalk_large_minus"]["kazakh"][set_name],
                          "turbo": get("turbo"), "small_ours": get("cleartalk"),
                          "pipelines": [get("pipeline:auto"), get("pipeline:mixed")] if set_name == "mixed" else None,
                          "units": ours["units"], "words": ours["words_auto" if set_name == "mixed" else "words_forced"]}
    return rows


def table(rows, language, label):
    head = HEAD[language]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    word = "слов" if language == "ru" else "words"
    for set_name in SETS:
        row = rows[set_name]
        pipelines = " / ".join(pct(value, language) for value in row["pipelines"]) if row["pipelines"] else "-"
        lines.append(f"| **{label}** · {NAMES[language][set_name]} ({row['units']}, {count(row['words'], language)} {word}) | {pct(row['start'], language)} | "
                     f"**{pct(row['ours'], language)}** | {points(row['compare'], language)} | {pct(row['turbo'], language)} | "
                     f"{pct(row['small_ours'], language)} | {pipelines} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ru", action="store_true")
    args = parser.parse_args()
    language = "ru" if args.ru else "en"
    sys.stdout.reconfigure(encoding="utf-8")
    final = test_rows()
    if final:
        print(table(final, language, "Итоговый тест B" if args.ru else "Final test B"))
        print()
    print(table(validation_rows(), language, "Валидация" if args.ru else "Validation"))


if __name__ == "__main__":
    main()
