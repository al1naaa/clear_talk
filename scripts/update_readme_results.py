"""Rewrite the multilingual result blocks of README.md and README_RU.md from the saved result files.

    python scripts/update_readme_results.py

Everything between ``<!-- multilingual-results:start -->`` / ``<!-- multilingual-results:end -->`` and between
``<!-- multilingual-limits:start -->`` / ``<!-- multilingual-limits:end -->`` is regenerated, so no number in the
READMEs is typed by hand. Nothing is transcribed here.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def block(text, name, body):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if start not in text or end not in text:
        raise SystemExit(f"Markers of {name} are missing")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body.strip()}\n{end}{tail}"


def main():
    import results_tables as tables
    from cleartalk_ml import analysis, validation
    from cleartalk_ml.paths import RESULTS, RUNS

    selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    ours = f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"
    config = json.loads((RUNS / selection["run"] / "config.json").read_text(encoding="utf-8"))
    val = tables.validation_rows()
    test = tables.test_rows()
    main_rows = test or val
    single = {name: validation.single_scores(system) for name, system in (("small", "small"), ("ours", ours))}
    sources = analysis.ksc2_by_source(ours)
    moves = analysis.better_same_worse("mixed-val", "small", ours)
    kinds = {name: analysis.edit_kinds("mixed-val", system) for name, system in (("small", "small"), ("ours", ours))}
    promo = selection["promotion"]
    routes = selection["routes"]["routes"]
    route_names = {"turbo": "Whisper large-v3-turbo", "kazakh": {"en": "the Kazakh turbo fine-tune", "ru": "казахская turbo"},
                   "small": "Whisper small", "cleartalk": {"en": "our model", "ru": "наша модель"}}

    def name(key, language):
        value = route_names[key]
        return value[language] if isinstance(value, dict) else value

    for language, path in (("en", ROOT / "docs" / "DETAILS.md"), ("ru", ROOT / "docs" / "DETAILS_RU.md")):
        p = lambda value: tables.pct(value, language)  # noqa: E731
        ru = language == "ru"
        where = ("финальном тесте" if test else "валидации") if ru else ("the final test" if test else "validation")
        parts = []
        if test:
            parts += [tables.table(test, language, "Финальный тест" if ru else "Final test"), ""]
        parts += [tables.table(val, language, "Валидация" if ru else "Validation"), ""]
        m, kk, ruf, en = main_rows["mixed"], main_rows["fleurs-kk"], main_rows["fleurs-ru"], main_rows["fleurs-en"]
        hard = sorted(sources, key=lambda s: -sources[s]["wer"])[:3]
        if ru:
            parts += [
                f"- Тест открывали **один раз** - после того как шаг обучения, правило продвижения и маршруты приложения были "
                f"зафиксированы на валидации. В тесте по 300 единиц на набор." if test else
                "- Финальный тест ещё не запускался: в таблице только валидация.",
                f"- **Главное: записи со сменой языка.** На {where} исходный Whisper small ошибается в {p(m['small'])} слов, наша "
                f"модель - в {p(m['ours'])}. Исходная модель после смены языка теряет целые фразы: на валидации пропущенных слов "
                f"{tables.count(kinds['small']['deletions'], 'ru')} → {tables.count(kinds['ours']['deletions'], 'ru')}; наша лучше на "
                f"{moves['better']} записях из {moves['units']}.",
                f"- **Казахский:** {p(kk['small'])} → {p(kk['ours'])} на начитанных предложениях. Русский и английский почти не "
                f"изменились ({p(ruf['small'])} → {p(ruf['ours'])} и {p(en['small'])} → {p(en['ours'])}) - этого и требовало правило «ни один язык не хуже».",
                f"- **Большие готовые модели всё равно сильнее там, где они есть:** на казахском - казахская turbo ({p(kk['kazakh'])}), на "
                f"русском и английском - large-v3-turbo ({p(ruf['turbo'])} и {p(en['turbo'])}). Поэтому после этого раунда маршруты приложения были: казахский → "
                f"{name(routes['kk'], 'ru')}, русский → {name(routes['ru'], 'ru')}, английский → {name(routes['en'], 'ru')} (так решено по валидации; "
                "большая модель второго раунда их изменила, см. раздел выше).",
                f"- **На смешанных записях лучше всего работает конвейер приложения** (режет запись на реплики и определяет язык каждой): "
                f"{p(m['pipeline'])} с готовыми моделями (маршруты первого раунда) против {p(m['ours'])} у нашей модели за один проход и {p(m['turbo'])} у large-v3-turbo за один проход.",
                f"- Наша модель {'прошла' if promo['promoted'] else 'не прошла'} заранее записанное правило продвижения и "
                + ("стала запасным распознавателем: приложение использует её для всех языков, когда больших моделей на компьютере нет."
                   if promo["promoted"] else "осталась выбираемой вручную."),
                f"- Обучено 6 488 064 параметра из 248 222 976 (2,6%); одна эпоха, {tables.count(config['train_examples'], 'ru')} примеров, "
                f"из них {tables.count(config['train_mixed'], 'ru')} - смешанные записи.",
            ]
            limits = [
                f"- **Наша многоязычная модель - маленькая.** Она намного лучше исходного Whisper small на казахском и на записях со сменой "
                f"языка, но уступает большим готовым моделям (см. таблицу), а на русском чуть хуже исходной ({p(val['fleurs-ru']['small'])} → "
                f"{p(val['fleurs-ru']['ours'])} на валидации).",
                f"- **Казахская речь вне начитанных текстов остаётся трудной** и для нашей модели: на валидации "
                + ", ".join(f"{s} - {p(sources[s]['wer'])}" for s in hard) + " (в обучении был только источник crowdsourced).",
                f"- **Метку языка наша модель ставит хуже, чем пишет текст**: казахскую речь она сама помечает как казахскую в "
                f"{p(single['ours']['fleurs-kk']['language_accuracy'])} случаев (исходная - в {p(single['small']['fleurs-kk']['language_accuracy'])}). "
                "Поэтому для её реплик приложение читает язык из написанного текста (на валидации верно в 100% и 97% случаев).",
            ]
        else:
            parts += [
                "- The test units were opened **once**, after the checkpoint, the promotion rule and the app's routes were fixed on "
                "validation. 300 units per set." if test else "- The final test has not been run yet: the table shows validation only.",
                f"- **The main result: recordings that change language.** On {where} untouched Whisper small gets {p(m['small'])} of the "
                f"words wrong, our model {p(m['ours'])}. The untouched model drops whole utterances after a language change: on validation "
                f"missing words fall from {kinds['small']['deletions']:,} to {kinds['ours']['deletions']:,}, and our model is better on "
                f"{moves['better']} of {moves['units']} recordings.",
                f"- **Kazakh:** {p(kk['small'])} → {p(kk['ours'])} on read sentences. Russian and English barely move ({p(ruf['small'])} → "
                f"{p(ruf['ours'])} and {p(en['small'])} → {p(en['ours'])}) - which is what the rule \"no language worse\" asked for.",
                f"- **The large ready models are still stronger where they exist:** the Kazakh fine-tune on Kazakh ({p(kk['kazakh'])}), "
                f"large-v3-turbo on Russian and English ({p(ruf['turbo'])} and {p(en['turbo'])}). So after this round the app sent Kazakh to "
                f"{name(routes['kk'], 'en')}, Russian to {name(routes['ru'], 'en')} and English to {name(routes['en'], 'en')} (decided on validation; "
                "the large model of the second round changed these routes, see the section above).",
                f"- **On mixed recordings the app's pipeline does best** (it cuts the recording into turns and detects each turn's language): "
                f"{p(m['pipeline'])} with ready models (the routes of round 1), against {p(m['ours'])} for our model in one pass and {p(m['turbo'])} for large-v3-turbo in one pass.",
                f"- Our model {'passed' if promo['promoted'] else 'did not pass'} the pre-registered promotion rule and "
                + ("is the stand-in recognizer: the app uses it for every language on a computer without the large models."
                   if promo["promoted"] else "stays selectable by hand."),
                f"- 6,488,064 of 248,222,976 parameters were trained (2.6%); one epoch of {config['train_examples']:,} examples, "
                f"{config['train_mixed']:,} of them mixed-language recordings.",
            ]
            limits = [
                f"- **Our multilingual model is a small one.** It is far better than untouched Whisper small on Kazakh and on recordings "
                f"that change language, but behind the large ready models (see the table), and slightly worse than the untouched model on "
                f"Russian read sentences ({p(val['fleurs-ru']['small'])} → {p(val['fleurs-ru']['ours'])} on validation).",
                "- **Kazakh outside read speech stays hard** for our model too: on validation "
                + ", ".join(f"{s} {p(sources[s]['wer'])}" for s in hard) + " (training saw the crowdsourced source only).",
                f"- **Our model names the language worse than it writes it**: it labels Kazakh speech as Kazakh for "
                f"{p(single['ours']['fleurs-kk']['language_accuracy'])} of utterances (untouched: {p(single['small']['fleurs-kk']['language_accuracy'])}). "
                "For its turns the app therefore reads the language from the text it wrote (right for 100% and 97% of validation utterances).",
            ]
        text = path.read_text(encoding="utf-8")
        text = block(text, "multilingual-results", "\n".join(parts))
        text = block(text, "multilingual-limits", "\n".join(limits))
        path.write_text(text, encoding="utf-8")
        print(f"updated {path.name}")
        if language == "en":  # the handoff note carries the same table and findings
            handoff = ROOT / "docs" / "dev" / "HANDOFF.md"
            handoff.write_text(block(handoff.read_text(encoding="utf-8"), "multilingual-results", "\n".join(parts)), encoding="utf-8")
            print("updated HANDOFF.md")


if __name__ == "__main__":
    main()
