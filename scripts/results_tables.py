"""Print the result tables of the multilingual model as Markdown, straight from the saved result files.

    python scripts/results_tables.py            # English
    python scripts/results_tables.py --ru       # Russian

Validation numbers come from the cached transcripts (data/ml/results/hyp), test numbers from the one-time
final test (data/ml/results/test-final.json). Nothing is transcribed here.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SETS = ["mixed", "fleurs-kk", "fleurs-ru", "fleurs-en", "ksc2-kk", "golos-ru"]
NAMES = {
    "en": {"mixed": "Mixed-language recordings", "fleurs-kk": "Kazakh · read sentences (FLEURS)", "fleurs-ru": "Russian · read sentences (FLEURS)",
           "fleurs-en": "English · read sentences (FLEURS)", "ksc2-kk": "Kazakh · broadcast and read speech (KSC2)",
           "golos-ru": "Russian · short phrases (Golos)"},
    "ru": {"mixed": "Смешанные записи (три языка)", "fleurs-kk": "Казахский · чтение предложений (FLEURS)",
           "fleurs-ru": "Русский · чтение предложений (FLEURS)", "fleurs-en": "Английский · чтение предложений (FLEURS)",
           "ksc2-kk": "Казахский · эфир и чтение (KSC2)", "golos-ru": "Русский · короткие фразы (Golos)"},
}
HEAD = {
    "en": ("Set", "Whisper small (untouched)", "**ClearTalk (LoRA)**", "Change (95% bootstrap)", "Whisper large-v3-turbo", "Kazakh turbo", "App pipeline (ready models)"),
    "ru": ("Набор", "Whisper small (исходный)", "**ClearTalk (LoRA)**", "Изменение (95% интервал)", "Whisper large-v3-turbo", "Казахская turbo", "Конвейер приложения (готовые модели)"),
}


def pct(value, language):
    if value is None:
        return "-"
    text = f"{100 * value:.1f}%"
    return text.replace(".", ",") if language == "ru" else text


def count(value, language):
    """12,345 in English, 12 345 in Russian."""
    text = f"{value:,}"
    return text.replace(",", " ") if language == "ru" else text


def points(block, language):
    numbers = [f"{100 * block[key]:+.1f}".replace("-", "-") for key in ("difference", "low", "high")]
    if language == "ru":
        numbers = [n.replace(".", ",") for n in numbers]
    unit = "п.п." if language == "ru" else "points"
    return f"{numbers[0]} {unit} ({numbers[1]} … {numbers[2]})"


def validation_rows():
    from cleartalk_ml import validation
    from cleartalk_ml.evaluate import bootstrap_difference, details, units_of
    from cleartalk_ml.paths import RESULTS

    selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    ours = f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"
    single = {name: validation.single_scores(system) for name, system in (("small", "small"), ("ours", ours), ("turbo", "turbo"), ("kazakh", "kazakh"))}
    mixed = {name: validation.mixed_scores(system) for name, system in (("small", "small"), ("ours", ours), ("turbo", "turbo"), ("pipeline", "pipeline:auto"))}
    rows = {}
    for set_name in SETS:
        if set_name == "mixed":
            get = lambda name: mixed[name]["wer"] if mixed.get(name) else None  # noqa: E731
            compare = bootstrap_difference(mixed["small"]["details"], mixed["ours"]["details"])
            words = mixed["ours"]["reference_words"]
            units = mixed["ours"]["units"]
        else:
            get = lambda name: single[name][set_name]["forced"]["wer"] if set_name in single.get(name, {}) else None  # noqa: E731
            eval_set = {"fleurs": "fleurs-val", "ksc2": "ksc2-val", "golos": "golos-val"}[set_name.split("-")[0]]
            group = units_of(eval_set, language=set_name.split("-")[1])
            a, _ = validation.hypotheses(eval_set, "small")
            b, _ = validation.hypotheses(eval_set, ours)
            compare = bootstrap_difference(details(group, a, "forced"), details(group, b, "forced"))
            words = single["ours"][set_name]["forced"]["reference_words"]
            units = single["ours"][set_name]["units"]
        rows[set_name] = {"small": get("small"), "ours": get("ours"), "compare": compare, "turbo": get("turbo"), "kazakh": get("kazakh"),
                          "pipeline": get("pipeline") if set_name == "mixed" else None, "units": units, "words": words}
    return rows


def test_rows():
    from cleartalk_ml.paths import RESULTS

    path = RESULTS / "test-final.json"
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

        ours = next(item for item in final["scores"] if item["system"] == "cleartalk" and item["set"] == set_name)
        rows[set_name] = {"small": get("small"), "ours": get("cleartalk"), "compare": final["cleartalk_vs_small"][set_name], "turbo": get("turbo"),
                          "kazakh": get("kazakh"), "pipeline": get("pipeline:auto"), "units": ours["units"],
                          "words": ours["words_auto" if set_name == "mixed" else "words_forced"]}
    return rows


def table(rows, language, label):
    head = HEAD[language]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    word = "слов" if language == "ru" else "words"
    for set_name in SETS:
        row = rows[set_name]
        lines.append(f"| **{label}** · {NAMES[language][set_name]} ({row['units']}, {count(row['words'], language)} {word}) | {pct(row['small'], language)} | "
                     f"**{pct(row['ours'], language)}** | {points(row['compare'], language)} | {pct(row['turbo'], language)} | "
                     f"{pct(row['kazakh'], language)} | {pct(row['pipeline'], language)} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ru", action="store_true")
    args = parser.parse_args()
    language = "ru" if args.ru else "en"
    sys.stdout.reconfigure(encoding="utf-8")
    final = test_rows()
    if final:
        print(table(final, language, "Финальный тест" if args.ru else "Final test"))
        print()
    print(table(validation_rows(), language, "Валидация" if args.ru else "Validation"))


if __name__ == "__main__":
    main()
