"""Rewrite the "noise, short and long recordings" block of README.md, README_RU.md and HANDOFF.md.

    python scripts/update_readme_robustness.py

The table comes from data/ml/results/robustness.json (python -m cleartalk_ml.robustness report): the app, language
detected per turn, on VALIDATION recordings with added noise. Nothing is transcribed here.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CONDITIONS = ("clean", "white-20", "white-10", "babble-10", "babble-5")
HEAD = {"en": ["Validation recordings", "clean", "white noise 20 dB", "white noise 10 dB", "babble 10 dB", "babble 5 dB"],
        "ru": ["Проверочные записи", "без шума", "белый шум 20 дБ", "белый шум 10 дБ", "голоса фоном 10 дБ", "голоса фоном 5 дБ"]}
ROWS = {"en": [("mixed", "Mixed-language recordings (20-30 s)"), ("long", "Conversations of 7 minutes"),
               ("short-kk", "Short phrases ≤ 4 s · Kazakh"), ("short-ru", "Short phrases · Russian"), ("short-en", "Short phrases · English")],
        "ru": [("mixed", "Смешанные записи (20-30 с)"), ("long", "Разговоры по 7 минут"), ("short-kk", "Короткие фразы ≤ 4 с · казахский"),
               ("short-ru", "Короткие фразы · русский"), ("short-en", "Короткие фразы · английский")]}


def block(text, name, body):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if start not in text or end not in text:
        raise SystemExit(f"Markers of {name} are missing")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body.strip()}\n{end}{tail}"


def main():
    import results_tables as base

    rows = json.loads((ROOT / "data" / "ml" / "results" / "robustness.json").read_text(encoding="utf-8"))
    value = {(r["condition"], r["set"]): r["wer"] for r in rows if r["setting"] == "auto"}
    for language, path in (("en", ROOT / "docs" / "DETAILS.md"), ("ru", ROOT / "docs" / "DETAILS_RU.md")):
        lines = ["| " + " | ".join(HEAD[language]) + " |", "|" + "---|" * len(HEAD[language])]
        for key, label in ROWS[language]:
            cells = [base.pct(value[(c, key)], language) if (c, key) in value else "-" for c in CONDITIONS]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        note = ("WER through the app (language detected per turn). Noise is added to validation recordings; \"babble\" is four other "
                "people talking at once; dB says how much louder the speech is than the noise. - = not measured.") if language == "en" else (
                "Доля ошибок в словах через приложение (язык определяется по репликам). Шум добавлен к проверочным записям; «голоса "
                "фоном» - четыре других человека говорят одновременно; дБ - насколько речь громче шума. - = не измерялось.")
        body = "\n".join(lines) + "\n\n" + note
        path.write_text(block(path.read_text(encoding="utf-8"), "robustness-results", body), encoding="utf-8")
        print("updated", path.name)
        if language == "en":
            handoff = ROOT / "docs" / "dev" / "HANDOFF.md"
            handoff.write_text(block(handoff.read_text(encoding="utf-8"), "robustness-results", body), encoding="utf-8")
            print("updated HANDOFF.md")


if __name__ == "__main__":
    main()
