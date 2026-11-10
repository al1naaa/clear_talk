"""Write the results of round 4 (noise, hallucinations, people talking at once) into the documents.

    python scripts/update_docs_round4.py

Every number is read from data/ml/results/. Blocks between markers are replaced, so the script can be run again.
README.md, README_RU.md, HANDOFF.md: the robustness block and a "protections" block; RESUME.md and
docs/PRESENTATION_RU.md: a round-4 section.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "ml" / "results"


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def en(value):
    return f"{100 * value:.1f}%"


def ru(value):
    return f"{100 * value:.1f}".replace(".", ",") + "%"


def diff(block, fmt=en):
    sign = lambda v: f"{100 * v:+.1f}".replace(".", "," if fmt is ru else ".")  # noqa: E731
    return f"{sign(block['difference'])} ({sign(block['low'])} … {sign(block['high'])})"


def replace_block(path, name, body, after=None):
    """Replace <!-- name:start --> … <!-- name:end --> in a file, or insert it after the line ``after``."""
    text = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    text = text.replace("\r\n", "\n")
    block = f"<!-- {name}:start -->\n{body.strip()}\n<!-- {name}:end -->"
    pattern = re.compile(rf"<!-- {re.escape(name)}:start -->.*?<!-- {re.escape(name)}:end -->", re.S)
    if pattern.search(text):
        text = pattern.sub(lambda m: block, text)
    elif after is not None:
        assert after in text, (path.name, after[:60])
        text = text.replace(after, after + "\n\n" + block, 1)
    else:
        raise SystemExit(f"{path.name}: no place for {name}")
    path.write_text(text.replace("\n", newline), encoding="utf-8")


def main():
    final, test_d = load("robustness-final.json"), load("test-final-D.json")
    background, region, overlap, kazakh = load("option-background.json"), load("option-region.json"), load("option-overlap-v2.json"), load("option-kk.json")
    overlap_v1, region_v2, region_v1, selection = load("option-overlap-v1.json"), load("option-region-v2.json"), load("by-region.json"), load("selection-round4.json")
    rows = {(r["condition"], r["set"]): r for r in final["rows"]}
    conditions = ["clean", "white-20", "white-10", "babble-10", "babble-5"]

    def table(names, sets, fmt, note):
        lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
        for set_name, label in sets:
            cells = []
            for c in conditions:
                r = rows.get((c, set_name))
                if r is None:
                    cells.append("-")
                elif fmt(r["before"]) == fmt(r["after"]):
                    cells.append(fmt(r["after"]))
                else:
                    cells.append(f"{fmt(r['before'])} → **{fmt(r['after'])}**" if r["after"] < r["before"] else f"{fmt(r['before'])} → {fmt(r['after'])}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n\n" + note

    overlap_on = bool(overlap and overlap["adopted"])
    cp = overlap["overlap_cpwer"] if overlap else None

    # ---------------------------------------------------------------- English (README, HANDOFF)
    table_en = table(["Validation recordings", "clean", "white noise 20 dB", "white noise 10 dB", "babble 10 dB", "babble 5 dB"],
                     [("mixed", "Mixed-language recordings (20-30 s)"), ("long", "Conversations of 7 minutes"), ("short-kk", "Short phrases ≤ 4 s · Kazakh"),
                      ("short-ru", "Short phrases · Russian"), ("short-en", "Short phrases · English")], en,
                     "WER through the app (language detected per turn) before → after the protections of round 4; **bold** = better, "
                     "one number = unchanged. Noise is added to validation recordings; \"babble\" is 3-5 other people talking; dB says how "
                     "much louder the speech is than the noise. - = not measured.")
    protections_en = [
        "**Repetition loops** are cut back to three occurrences (a recogniser stuck in a loop wrote \"Senate\" 213 times); every mode.",
        f"**Background voices.** A turn of at most 3 s that is at least 5 dB quieter than the conversation and at most 10 dB above the "
        f"background is set aside: listed under the transcript, left out of the text and the exports. Validation {en(background['all']['before'])} → "
        f"{en(background['all']['after'])}, {diff(background['all'])}; babble 10 dB {en(background['by_condition']['babble-10']['before'])} → "
        f"{en(background['by_condition']['babble-10']['after'])}; clean unchanged.",
        f"**Whole stretches of speech.** The recogniser gets each stretch of continuous speech whole (a breath pause of one person no longer "
        f"cuts the sentence) and its sentences are given to speakers by time. Validation {en(region['all']['before'])} → {en(region['all']['after'])}, "
        f"{diff(region['all'])}; clean 7-minute conversations {en(region['clean_guards']['long']['before'])} → {en(region['clean_guards']['long']['after'])}.",
    ]
    if overlap:
        protections_en.append(
            (f"**People talking at once** are separated where the speaker labels see two people: on held-out overlap recordings cpWER "
             f"{en(cp['before'])} → {en(cp['after'])}, {diff(cp)}." if overlap_on else
             f"**People talking at once** - separating them was measured and **not adopted**: on held-out overlap recordings the error fell "
             f"{en(cp['before'])} → {en(cp['after'])}, but in ordinary and noisy conversations it invented words - one voice spread over two "
             f"tracks, background talkers written as a second voice (robustness recordings {en(overlap['all']['before'])} → {en(overlap['all']['after'])})."))
    rejected_en = (f"Not adopted (each by its pre-registered rule): training the large model in noise (noisy validation "
                   f"{en(selection['baseline_noisy_wer'])} → {en(selection['baseline_noisy_wer'] + selection['noisy']['difference'])}), two earlier "
                   f"variants of whole-stretch transcription, " + ("the first overlap variant, " if overlap_v1 else "")
                   + f"and the narrower Kazakh cross-check ({diff(kazakh['all'])}). Details: `ml_pipeline/round4/PIPELINE.md`.")
    if test_d and test_d.get("mode") == "app":
        by = test_d["condition"]
        test_en = ("**Final test D** (once; 898 utterances no system had seen, each clean, with white noise 10 dB and with babble 10 dB; "
                   "one person per recording): the app before → after the protections: "
                   + ", ".join(f"{name} {en(by[c]['before'])} → {en(by[c]['final'])} ({diff(by[c])})" for c, name in
                               (("clean", "clean"), ("white-10", "white noise 10 dB"), ("babble-10", "babble 10 dB")) if c in by)
                   + f"; all {en(test_d['all']['before'])} → {en(test_d['all']['final'])}.")
    else:
        test_en = "Final test D: not run yet."
    robust_en = table_en + "\n\n" + "\n".join(f"- {line}" for line in protections_en) + "\n\n" + rejected_en + "\n\n" + test_en
    for name in ("docs/DETAILS.md", "docs/dev/HANDOFF.md"):
        replace_block(ROOT / name, "robustness-results", robust_en)

    # ---------------------------------------------------------------- Russian (README_RU)
    table_ru = table(["Проверочные записи", "без шума", "белый шум 20 дБ", "белый шум 10 дБ", "голоса фоном 10 дБ", "голоса фоном 5 дБ"],
                     [("mixed", "Записи со сменой языка (20-30 с)"), ("long", "Разговоры по 7 минут"), ("short-kk", "Короткие фразы ≤ 4 с · казахский"),
                      ("short-ru", "Короткие фразы · русский"), ("short-en", "Короткие фразы · английский")], ru,
                     "Доля ошибок в словах через приложение (язык определяется для каждой реплики): до → после защит четвёртого раунда; "
                     "**жирным** - где стало лучше, одно число - без изменений. Шум добавлен к проверочным записям; «голоса фоном» - 3-5 "
                     "человек говорят на фоне; дБ - насколько речь громче шума. - = не измерялось.")
    protections_ru = [
        "**Зацикливание** обрезается: слово или фраза, повторённые больше трёх раз подряд, остаются три раза (модель однажды написала «Senate» 213 раз); во всех режимах.",
        f"**Фоновые голоса.** Реплика не длиннее 3 с, которая минимум на 5 дБ тише разговора и не больше чем на 10 дБ громче фона, "
        f"откладывается: она показана бледным списком под расшифровкой и не входит в текст и файлы. Проверка: {ru(background['all']['before'])} → "
        f"{ru(background['all']['after'])}, {diff(background['all'], ru)} п.; голоса фоном 10 дБ {ru(background['by_condition']['babble-10']['before'])} → "
        f"{ru(background['by_condition']['babble-10']['after'])}; чистые записи без изменений.",
        f"**Сплошная речь целиком.** Модель получает каждый кусок сплошной речи целиком (пауза на вдох внутри речи одного человека больше "
        f"не режет фразу), а предложения раздаются говорящим по времени. Проверка: {ru(region['all']['before'])} → {ru(region['all']['after'])}, "
        f"{diff(region['all'], ru)} п.; чистые разговоры по 7 минут {ru(region['clean_guards']['long']['before'])} → {ru(region['clean_guards']['long']['after'])}.",
    ]
    if overlap:
        protections_ru.append(
            (f"**Двое одновременно** разделяются там, где метки говорящих видят двух людей: на отложенных записях с наложением "
             f"{ru(cp['before'])} → {ru(cp['after'])}, {diff(cp, ru)} п." if overlap_on else
             f"**Двое одновременно** - разделение проверено и **не включено**: на записях с наложением ошибок стало {ru(cp['before'])} → "
             f"{ru(cp['after'])}, но на обычных и шумных разговорах появлялся лишний текст: один голос раскладывался на две дорожки, "
             f"а люди на фоне записывались как второй голос (проверочные записи {ru(overlap['all']['before'])} → {ru(overlap['all']['after'])})."))
    rejected_ru = (f"Не включено (каждое по своему правилу, записанному заранее): дообучение большой модели на шуме (шумная проверка "
                   f"{ru(selection['baseline_noisy_wer'])} → {ru(selection['baseline_noisy_wer'] + selection['noisy']['difference'])}), два первых "
                   f"варианта распознавания целыми кусками, " + ("первый вариант разделения наложений, " if overlap_v1 else "")
                   + f"узкая казахская перепроверка ({diff(kazakh['all'], ru)} п.). Подробно: `ml_pipeline/round4/PIPELINE.md`.")
    if test_d and test_d.get("mode") == "app":
        by = test_d["condition"]
        test_ru = ("**Итоговый тест D** (один раз; 898 фраз, которых не видела ни одна модель, каждая без шума, с белым шумом 10 дБ и с "
                   "голосами фоном 10 дБ; в записи говорит один человек): приложение до → после защит: "
                   + ", ".join(f"{name} {ru(by[c]['before'])} → {ru(by[c]['final'])} ({diff(by[c], ru)} п.)" for c, name in
                               (("clean", "без шума"), ("white-10", "белый шум 10 дБ"), ("babble-10", "голоса фоном 10 дБ")) if c in by)
                   + f"; всего {ru(test_d['all']['before'])} → {ru(test_d['all']['final'])}.")
    else:
        test_ru = "Итоговый тест D ещё не запускался."
    robust_ru = table_ru + "\n\n" + "\n".join(f"- {line}" for line in protections_ru) + "\n\n" + rejected_ru + "\n\n" + test_ru
    replace_block(ROOT / "docs" / "DETAILS_RU.md", "robustness-results", robust_ru)

    # ---------------------------------------------------------------- how Conversation mode works: the protections step
    step_en = ("6. **Protections against invented text** (round 4, each adopted on validation by a rule written before measuring): "
               "repetition loops are cut back; short quiet turns near the background level are set aside (listed under the "
               "transcript); every stretch of continuous speech goes to the recogniser whole and its sentences are given to speakers "
               "by time" + ("; where the speaker labels see two people at once, the two voices are separated and written apart." if overlap_on else
                            ". Separating people who talk at the same time was measured and not adopted (it invented words in ordinary conversations)."))
    step_ru = ("6. **Защиты от выдуманного текста** (четвёртый раунд, каждая принята по проверочным данным и правилу, записанному до замера): "
               "зацикливание обрезается; короткие тихие реплики на уровне фона откладываются (показаны под расшифровкой); каждый кусок "
               "сплошной речи распознаётся целиком, а предложения раздаются говорящим по времени"
               + ("; где метки говорящих видят двух людей одновременно, голоса разделяются и записываются отдельно." if overlap_on else
                  ". Разделение одновременной речи проверено и не включено: на обычных разговорах оно добавляло лишний текст."))
    replace_block(ROOT / "docs" / "DETAILS.md", "protections", step_en, after="   with any routes.")
    replace_block(ROOT / "docs" / "DETAILS_RU.md", "protections", step_ru, after="   Поэтому карты на 6 ГБ хватает при любых маршрутах.")

    # ---------------------------------------------------------------- limits
    if not overlap_on:
        limit_en = ("- **Conversation mode does not separate people who talk at the same time**; overlapping words can be lost or given to one "
                    "speaker. A separation step was built and measured (round 4) but not adopted: it halved the errors on overlap recordings and "
                    "invented words in ordinary conversations. The English two-voice mode exists for exactly two simultaneous voices.\n"
                    "- **Set-aside background turns are worth a look**: a quiet real reply in a noisy room can land among them.")
        limit_ru = ("- **Люди, говорящие одновременно, в режиме «Разговор» не разделяются**: слова могут потеряться или достаться одному "
                    "говорящему. Разделение было сделано и проверено (четвёртый раунд), но не включено: на записях с наложением оно вдвое "
                    "уменьшало ошибки, а на обычных разговорах добавляло лишний текст. Для двух одновременных голосов есть английский режим.\n"
                    "- **Отложенные «фоновые» реплики стоит просмотреть**: тихий настоящий ответ в шумной комнате может оказаться среди них.")
    else:
        limit_en = ("- **People talking at once are separated only where the speaker labels see two people**; elsewhere overlapping words can "
                    "be lost or given to one speaker. Separated turns are marked for checking.\n"
                    "- **Set-aside background turns are worth a look**: a quiet real reply in a noisy room can land among them.")
        limit_ru = ("- **Одновременная речь разделяется, только если метки говорящих видят там двух людей**; в остальных местах слова могут "
                    "потеряться или достаться одному говорящему. Разделённые реплики помечены «проверить».\n"
                    "- **Отложенные «фоновые» реплики стоит просмотреть**: тихий настоящий ответ в шумной комнате может оказаться среди них.")
    for path, old, body in ((ROOT / "docs" / "DETAILS.md", "- **Conversation mode does not separate people who talk at the same time**; overlapping words can be lost or\n"
                             "  given to one speaker. The English two-voice mode exists for exactly two simultaneous voices.", limit_en),
                            (ROOT / "docs" / "DETAILS_RU.md", "- **Люди, говорящие одновременно, в режиме «Разговор» не разделяются**: слова могут потеряться или\n"
                             "  достаться одному говорящему. Для двух одновременных голосов есть отдельный английский режим.", limit_ru)):
        text = path.read_text(encoding="utf-8")
        newline = "\r\n" if "\r\n" in text else "\n"
        text = text.replace("\r\n", "\n")
        if old in text:
            text = text.replace(old, f"<!-- overlap-limits:start -->\n{body}\n<!-- overlap-limits:end -->", 1)
            path.write_text(text.replace("\n", newline), encoding="utf-8")
        else:
            replace_block(path, "overlap-limits", body)
    # ---------------------------------------------------------------- the presentation script
    if test_d and test_d.get("mode") == "app":
        by = test_d["condition"]
        test_line = ("Итоговый тест D (898 фраз, которых никто не видел, открыт один раз): "
                     + ", ".join(f"{name} {ru(by[c]['before'])} → {ru(by[c]['final'])}" for c, name in
                                 (("clean", "без шума"), ("white-10", "белый шум 10 дБ"), ("babble-10", "голоса фоном 10 дБ")) if c in by) + ".")
    else:
        test_line = "Итоговый тест D ещё не запускался."
    noisy_mixed, long_babble, long_clean = rows[("babble-10", "mixed")], rows[("babble-10", "long")], rows[("clean", "long")]
    presentation = f"""## Дополнение: защиты от выдуманного текста (четвёртый раунд)

Все цифры - из `ml_pipeline/round4/PIPELINE.md` и файлов `data/ml/results/option-*.json`, `robustness-final.json`,
`test-final-D.json`.

**Одна фраза.**

> Мы разобрали, где программа выдумывает текст в шуме, и добавили три защиты: обрезаем зацикливание, откладываем
> короткие тихие кусочки фоновых голосов и распознаём сплошную речь целиком. На записях с фоновыми голосами ошибок
> стало {ru(noisy_mixed['before'])} → {ru(noisy_mixed['after'])}, на разговорах - {ru(long_babble['before'])} → {ru(long_babble['after'])}; чистые записи не пострадали.

**Слайды 9 и 10.** Таблица «до → после» и карточки защит. {test_line}

**Что было и что сделали.**
- *Зацикливание:* модель иногда пишет одно слово десятки раз («Senate» 213 раз). Повтор больше трёх раз подряд обрезается.
- *Фоновые голоса:* короткие кусочки, где слышен только фон, программа считала новым «говорящим» и писала бессмыслицу
  («Эпизод первого сезона»). Теперь кусок до 3 с, который заметно тише разговора и почти на уровне фона, откладывается -
  его видно бледным списком под расшифровкой, но в текст и файлы он не входит. Пороги подобраны на половине проверочных
  записей и подтверждены на другой половине.
- *Сплошная речь целиком:* раньше фраза резалась там, где метки говорящих «мигали», и на стыках слова терялись.
  Теперь кусок сплошной речи распознаётся целиком, а предложения раздаются говорящим по времени. Чистые разговоры:
  {ru(long_clean['before'])} → {ru(long_clean['after'])}.

**Что честно не получилось.**
- Дообучение большой модели на шуме не дало выигрыша (26,3% → 26,7% на шумной проверке) - оставили модель второго раунда.
- Разделение двух одновременных голосов в «Разговоре»: на записях, где двое говорят одновременно, ошибок вдвое меньше
  ({ru(cp['before'])} → {ru(cp['after'])}), но на обычных и шумных разговорах оно записывало лишнее (один голос на двух
  дорожках, люди на фоне как «второй голос») - правило его не пропустило, и оно выключено.
- Распознавание целыми кусками получилось с третьей попытки; первые две не прошли правило, и это записано.

**Что показать.** В режиме «Разговор» выбрать файл `data/demo/conversation_cafe.wav` - тот же пример «Три языка.
Шесть голосов.», но с людьми, говорящими на фоне (как в кафе). Приложение пишет шесть ровных реплик; программа до этих
защит на этой же записи придумала одну казахскую фразу из фонового шума и разрезала две фразы (24,0% → 10,7% ошибок).
На Mac расшифровку лучше запустить заранее (около 3 минут на процессоре).

**Вопросы, которые стоит ждать.**

*«Галлюцинаций теперь нет совсем?»* - Нет, и так говорить нельзя. Их стало заметно меньше в шуме, а сомнительные места
помечены «Проверьте». Совсем убрать выдумки у моделей такого типа невозможно; можно уменьшить и показать честно.

*«Не подогнали ли пороги под проверку?»* - Каждое правило записано в журнал до замера. Пороги фоновых кусочков подобраны
на одной половине записей, проверены на другой, а итог - на тестовом наборе, который открыли один раз.

*«Что, если настоящий ответ тихий?»* - Он не исчезает: отложенные кусочки показаны под расшифровкой, их можно просмотреть.
На чистых записях ни одна реплика не была отложена."""
    path = ROOT / "docs" / "PRESENTATION_RU.md"
    text = path.read_text(encoding="utf-8")
    marker = "## Что важно сказать самой"
    if "<!-- round4:start -->" in text:
        replace_block(path, "round4", presentation)
    else:
        text = text.replace(marker, f"<!-- round4:start -->\n{presentation}\n<!-- round4:end -->\n\n{marker}", 1)
        path.write_text(text, encoding="utf-8")
    print("documents updated")


if __name__ == "__main__":
    main()
