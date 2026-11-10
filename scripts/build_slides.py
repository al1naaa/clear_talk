"""The slides for the defense, in Russian: docs/ClearTalk_defense_RU.pptx (and a PNG of every slide for checking).

    python scripts/build_slides.py

Every number on the slides is read from the saved result files, so the deck can be rebuilt after any new
measurement. The deck itself is drawn by PowerPoint (scripts/render_slides.ps1, through COM), so building it needs
PowerPoint on this computer; the .pptx then opens in PowerPoint or Keynote anywhere. Speaker notes are included.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "ml" / "results"
OUT = ROOT / "docs" / "ClearTalk_defense_RU.pptx"
SPEC = ROOT / "docs" / "slides" / "deck.json"
PNG = ROOT / "docs" / "slides" / "png"

DARK, GREEN, CARD, CARD2 = "1F3A2E", "2F4A3A", "EEF3EC", "F6F8F4"
ORANGE, TEXT, MUTED, WHITE, SOFT = "DE693C", "1E2A24", "5F6F66", "FFFFFF", "CADBCF"
KK, RU, EN, GREY = "2A9D8F", "3B6EA8", "D08C2B", "A3ADA6"
W, H, M = 960, 540, 48


def pct(value, digits=1):
    return f"{100 * value:.{digits}f}".replace(".", ",") + "%"


def title(text, color=TEXT):
    return {"type": "text", "x": M, "y": 32, "w": W - 2 * M, "h": 50, "text": text, "size": 30, "bold": True,
            "font": "Cambria", "color": color, "valign": "middle", "name": "title"}


def text(x, y, w, h, value, size=16, color=TEXT, **extra):
    return {"type": "text", "x": x, "y": y, "w": w, "h": h, "text": value, "size": size, "color": color, **extra}


def box(x, y, w, h, fill, **extra):
    return {"type": "box", "x": x, "y": y, "w": w, "h": h, "fill": fill, **extra}


def circle(x, y, d, fill, label="", size=16, color=WHITE):
    return {"type": "box", "shape": "circle", "x": x, "y": y, "w": d, "h": d, "fill": fill, "label": label, "size": size,
            "color": color, "bold": True}


def results():
    test_b = json.loads((RESULTS / "test-final-B.json").read_text(encoding="utf-8"))
    score = {(item["system"], item["set"]): item["wer_auto"] if item["set"] == "mixed" else item["wer_forced"] for item in test_b["scores"]}
    english = {item["set"]: item for item in json.loads((ROOT / "data" / "asr" / "results" / "final-summary.json").read_text(encoding="utf-8"))["test"]}
    robust = RESULTS / "robustness.json"
    return score, english, (json.loads(robust.read_text(encoding="utf-8")) if robust.is_file() else None), test_b


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def loop_effect(condition="babble-10", set_name="short-en"):
    """WER of the app's stored validation transcripts without and with the loop cut (a pure text step)."""
    sys.path.insert(0, str(ROOT))
    from cleartalk_ml import robustness
    from cleartalk_ml.evaluate import details, pick_device, read_hypotheses, without_loops
    from cleartalk_ml.text import pooled

    device, precision = pick_device()
    plan = json.loads(robustness.PLAN.read_text(encoding="utf-8"))
    units = robustness.units(plan, condition)
    group = units[units.set == set_name]
    rows = read_hypotheses(robustness.hyp_file(condition, "auto", f"{device}-{precision}"))
    kind = "auto" if set_name in ("mixed", "long") else "forced"
    return pooled(details(group, rows, kind).to_dict("records"))["wer"], pooled(details(group, without_loops(rows), kind).to_dict("records"))["wer"]


def robustness_final_slide(final):
    names = {"clean": "без шума", "white-20": "шум 20 дБ", "white-10": "шум 10 дБ", "babble-10": "голоса 10 дБ", "babble-5": "голоса 5 дБ"}
    sets = [("mixed", "Смешанные записи (20-30 с)"), ("short-kk", "Короткие фразы ≤ 4 с, казахский"), ("short-ru", "Короткие фразы, русский"),
            ("short-en", "Короткие фразы, английский"), ("long", "Длинные разговоры, 7 минут")]
    lookup = {(row["condition"], row["set"]): row for row in final["rows"]}
    rows = [["Записи (проверочные)"] + list(names.values())]
    for set_name, label in sets:
        cells = []
        for condition in names:
            row = lookup.get((condition, set_name))
            if row is None:
                cells.append("-")
            else:
                before, after = pct(row["before"]), pct(row["after"])
                cells.append(before if before == after else f"{before} → **{after}" if row["after"] < row["before"] else f"{before} → {after}")
        rows.append([label] + cells)
    elements = [
        title("Шум, короткие и длинные записи: до и после"),
        {"type": "table", "x": M, "y": 100, "w": W - 2 * M, "rows": rows, "col_widths": [244] + [(W - 2 * M - 244) / 5] * 5, "size": 12,
         "row_height": 40, "header_fill": GREEN, "header_color": WHITE, "band_fill": CARD2, "strong_color": ORANGE},
        text(M, 100 + 40 * len(rows) + 14, W - 2 * M, 70,
             "Доля ошибок в словах через приложение (язык определяется сам), проверочные записи с добавленным шумом: «шум» - белый шум, "
             "«голоса» - 3-5 человек говорят фоном; дБ - насколько речь громче шума. Слева - приложение до защит этого раунда, "
             "справа - с ними; оранжевым - где стало лучше, одно число - без изменений.", 12, MUTED),
    ]
    return {"elements": elements, "notes": "Все записи проверочные. Чистые записи почти не изменились - защиты срабатывают там, где есть шум, "
                                           "фоновые голоса или паузы внутри фраз. Тест на нетронутых данных - на следующем слайде."}


def protections_slide(options, test_d):
    loops_before, loops_after = loop_effect()
    background, region, overlap, kazakh = (options.get(k) for k in ("background", "region", "overlap", "kk"))
    cards = [("Зацикливание", "Модель иногда пишет одно слово десятки раз («Senate Senate Senate…»). Повтор больше трёх раз подряд обрезается.",
              f"{pct(loops_before)} → {pct(loops_after)}", "короткие английские фразы при фоновых голосах")]
    if background and background["adopted"]:
        cond = background["by_condition"]["babble-10"]
        cards.append(("Фоновые голоса", "Короткий тихий кусок, где слышен только фон, откладывается: он показан бледным списком, но в текст не входит.",
                      f"{pct(cond['before'])} → {pct(cond['after'])}", "все записи с фоновыми голосами 10 дБ"))
    if region and region["adopted"]:
        guard = region["clean_guards"]["long"]
        cards.append(("Паузы не режут фразу", "Сплошная речь распознаётся целиком, пауза на вдох внутри речи одного человека не разрывает фразу; "
                      "слова раздаются говорящим по времени.", f"{pct(guard['before'])} → {pct(guard['after'])}", "чистые разговоры по 7 минут"))
    if overlap:
        cp = overlap["overlap_cpwer"]
        if overlap["adopted"]:
            cards.append(("Двое одновременно", "Где метки показывают двух людей сразу, голоса разделяются и каждый записывается отдельно.",
                          f"{pct(cp['before'])} → {pct(cp['after'])}", "ошибок на записях с наложением голосов"))
        else:
            cards.append(("Двое одновременно", "Разделение голосов в местах наложения проверено, но не включено: не прошло правило, записанное заранее.",
                          f"{pct(cp['before'])} → {pct(cp['after'])}", "ошибок на записях с наложением (проверка)"))
    if kazakh and kazakh["adopted"]:
        cards.append(("Казахская фраза ≠ английская", "Если детектор принял казахскую фразу за английскую, а наша модель услышала казахский, остаётся её текст.",
                      f"{pct(kazakh['all']['before'])} → {pct(kazakh['all']['after'])}", "смешанные записи и короткие фразы"))
    cards = cards[:4]
    elements = [title("Без выдуманного текста: защиты")]
    card_w, card_h = (W - 2 * M - 24) / 2, 158
    for i, (head, body, number, label) in enumerate(cards):
        x, y = M + (i % 2) * (card_w + 24), 100 + (i // 2) * (card_h + 14)
        elements += [box(x, y, card_w, card_h, CARD), text(x + 18, y + 10, card_w - 36, 28, head, 18, TEXT, bold=True),
                     text(x + 18, y + 40, card_w - 200, 110, body, 12, TEXT),
                     text(x + card_w - 178, y + 44, 164, 40, number, 20, ORANGE, bold=True, font="Cambria", align="right"),
                     text(x + card_w - 178, y + 88, 164, 60, label, 11, MUTED, align="right")]
    if test_d and test_d.get("mode") == "app":
        by = test_d["condition"]
        line = ("Тест на данных, которых никто не видел (одна фраза - один человек): ошибок "
                + ", ".join(f"{name} {pct(by[c]['before'])} → {pct(by[c]['final'])}" for c, name in
                            (("clean", "без шума"), ("white-10", "шум 10 дБ"), ("babble-10", "фоновые голоса 10 дБ")) if c in by) + ".")
    else:
        line = "Каждая защита включена, только если прошла правило, записанное до проверки; на чистых записях ни одна не ухудшила результат больше допуска."
    elements.append(text(M, 100 + 2 * (card_h + 14) + 4, W - 2 * M, 50, line, 13, TEXT, bold=True))
    notes = ("Каждая защита сначала проверялась на проверочных записях по правилу, записанному до замера. "
             "Не прошли и честно отброшены: обучение большой модели на шуме и два первых варианта распознавания целыми кусками.")
    return {"elements": elements, "notes": notes}


def robustness_slide(robust):
    names = {"clean": "без шума", "white-20": "шум 20 дБ", "white-10": "шум 10 дБ", "babble-10": "голоса 10 дБ", "babble-5": "голоса 5 дБ"}
    sets = [("mixed", "auto", "Смешанные записи (20-30 с)"), ("mixed", "mixed", "  то же, «языки меняются в реплике»"),
            ("short-kk", "auto", "Короткие фразы ≤ 4 с, казахский"), ("short-ru", "auto", "Короткие фразы, русский"),
            ("short-en", "auto", "Короткие фразы, английский"), ("long", "auto", "Длинные разговоры, 7 минут"),
            ("long", "mixed", "  то же, «языки меняются в реплике»")]
    elements = [title("Шум, короткие и длинные записи")]
    if robust is None:
        elements.append(text(M, 120, W - 2 * M, 60, "Проверка на зашумлённых, коротких и длинных записях ещё идёт - таблица появится здесь.", 18, MUTED))
        return {"elements": elements, "notes": ""}
    lookup = {(row["condition"], row["setting"], row["set"]): row["wer"] for row in robust}
    header = ["Записи (проверочные)"] + [names[c] for c in names]
    rows = [header]
    for set_name, setting, label in sets:
        values = [lookup.get((condition, setting, set_name)) for condition in names]
        if all(v is None for v in values):
            continue
        rows.append([label] + ["-" if v is None else pct(v) for v in values])
    elements += [
        {"type": "table", "x": M, "y": 100, "w": W - 2 * M, "rows": rows, "col_widths": [270] + [(W - 2 * M - 270) / 5] * 5, "size": 13,
         "row_height": 34, "header_fill": GREEN, "header_color": WHITE, "band_fill": CARD2},
        text(M, 100 + 34 * len(rows) + 16, W - 2 * M, 60,
             "Доля ошибок в словах через приложение. Шум добавлен к проверочным записям: «шум» - белый шум, «голоса фоном» - четыре "
             "человека говорят одновременно; дБ - насколько речь громче шума (меньше - тяжелее).", 13, MUTED),
    ]
    return {"elements": elements, "notes": "Эта таблица отвечает на вопрос, как приложение ведёт себя в шуме и на записях разной длины. "
                                           "Все записи - проверочные, к ним искусственно добавлен шум."}


def build():
    score, english, robust, test_b = results()
    t1, t2, t3 = english["T1 · v4 test mixtures"], english["T2 · new test mixtures"], english["T3 · clean single speaker"]
    english_bars = []
    for label, item in (("Новые смеси двух голосов", t2), ("Смеси раннего эксперимента", t1), ("Чистая речь одного человека", t3)):
        english_bars += [{"label": f"{label} · готовая", "value": item["pretrained_wer"], "text": pct(item["pretrained_wer"]), "color": GREY},
                         {"label": "наша", "value": item["cleartalk_wer"], "text": pct(item["cleartalk_wer"]), "color": ORANGE, "bold": True}]
    slides = []

    # 1 - title
    slides.append({"bg": DARK, "elements": [
        text(M, 70, 600, 24, "ПРОЕКТ ПО ОБРАБОТКЕ ЕСТЕСТВЕННОГО ЯЗЫКА", 13, ORANGE, bold=True),
        text(M, 110, 700, 80, "ClearTalk", 60, WHITE, bold=True, font="Cambria"),
        text(M, 200, 560, 90, "Распознавание речи, когда говорят несколько человек - на казахском, русском и английском", 22, SOFT),
        circle(650, 120, 90, KK, "Қазақша", 15), circle(745, 200, 90, RU, "Русский", 15), circle(650, 280, 90, EN, "English", 15),
        text(M, 440, 860, 40, "Три модели распознавания, дообученные на ноутбуке · приложение для разговоров и двух голосов", 15, SOFT),
    ], "notes": "Одна фраза о проекте: ClearTalk превращает запись разговора в текст - находит реплики, говорящих и язык каждой "
                "реплики. Мы дообучили три модели распознавания речи Whisper методом LoRA."})

    # 2 - the problem
    slides.append({"elements": [
        title("Где обычное распознавание речи ломается"),
        box(M, 110, 420, 330, CARD), circle(M + 24, 134, 44, ORANGE, "1", 18),
        text(M + 84, 136, 316, 44, "Двое говорят одновременно", 21, TEXT, bold=True, valign="middle"),
        text(M + 24, 200, 372, 90, "Слова двух людей смешиваются в один бессвязный текст - типичная ситуация на встречах и интервью.", 16, TEXT),
        text(M + 24, 310, 372, 60, pct(t2["pretrained_wer"], 0), 44, ORANGE, bold=True, font="Cambria"),
        text(M + 24, 370, 372, 50, "ошибок в словах у готового Whisper на смесях двух голосов", 14, MUTED),
        box(W - M - 420, 110, 420, 330, CARD), circle(W - M - 396, 134, 44, ORANGE, "2", 18),
        text(W - M - 336, 136, 316, 44, "Язык меняется в разговоре", 21, TEXT, bold=True, valign="middle"),
        text(W - M - 396, 200, 372, 90, "В одном разговоре в Казахстане часто звучат казахский, русский и английский; модель продолжает писать на первом языке или теряет фразы.", 16, TEXT),
        text(W - M - 396, 310, 372, 60, pct(score[("small", "mixed")], 0), 44, ORANGE, bold=True, font="Cambria"),
        text(W - M - 396, 370, 372, 50, "ошибок у Whisper small на записях со сменой языка", 14, MUTED),
    ], "notes": "Две проблемы, которые мы решали. Обе цифры - из итоговых тестов на записях, которых модели не видели."})

    # 3 - the app
    steps = [("Поиск речи", "детектор речи Silero"), ("Кто говорит", "голосовые «отпечатки» + группировка"),
             ("Язык реплики", "Whisper, только kk / ru / en"), ("Модель для языка", "выбрана по проверке"),
             ("Текст", "имена, исправления, субтитры")]
    elements = [title("Приложение: запись → текст с говорящими и языками")]
    step_w, gap = 150, 26
    for i, (head, caption) in enumerate(steps):
        x = M + i * (step_w + gap)
        elements += [box(x, 130, step_w, 70, GREEN if i < 4 else ORANGE, label=head, size=16),
                     text(x, 212, step_w, 60, caption, 13, MUTED, align="center")]
        if i < 4:
            elements.append({"type": "arrow", "x1": x + step_w + 3, "y1": 165, "x2": x + step_w + gap - 3, "y2": 165, "color": MUTED})
    elements += [
        box(M, 300, W - 2 * M, 150, CARD2),
        text(M + 24, 318, W - 2 * M - 48, 30, "Два режима", 18, TEXT, bold=True),
        {"type": "bullets", "x": M + 24, "y": 352, "w": W - 2 * M - 48, "h": 90, "size": 15, "space_after": 6, "bullet_color": ORANGE, "items": [
            "Разговор: несколько человек по очереди, казахский / русский / английский, записи до 60 минут",
            "Два голоса одновременно (английский): SepFormer разделяет смесь, наш Whisper small.en пишет каждый голос",
            "Интерфейс на русском и английском, запись с микрофона, сохранение .txt / .srt / .vtt"]},
    ]
    slides.append({"elements": elements, "notes": "Так устроен режим «Разговор». Готовые модели используются как есть, а там, "
                                                  "где наша модель оказалась лучше на проверке, приложение отдаёт реплику ей."})

    # 4 - three trained models
    cards = [("Английская", "Whisper small.en + LoRA", "Для голосов после разделителя.\n6,5 млн обученных параметров (2,6%)"),
             ("Многоязычная малая", "Whisper small + LoRA", "Казахский, русский, английский и смена языка.\n6,5 млн параметров (2,6%)"),
             ("Многоязычная большая", "Казахская Whisper large-v3-turbo + LoRA в декодере", "Обучена только часть, которая пишет текст.\n2,1 млн из 811 млн (0,3%)")]
    elements = [title("Наш вклад: три дообученные модели")]
    card_w = (W - 2 * M - 2 * 24) / 3
    for i, (head, model, body) in enumerate(cards):
        x = M + i * (card_w + 24)
        elements += [box(x, 110, card_w, 270, CARD), circle(x + 20, 130, 44, ORANGE, str(i + 1), 18),
                     text(x + 20, 190, card_w - 40, 30, head, 20, TEXT, bold=True),
                     text(x + 20, 224, card_w - 40, 46, model, 15, GREEN, bold=True),
                     text(x + 20, 280, card_w - 40, 90, body, 14, TEXT)]
    elements.append(text(M, 410, W - 2 * M, 40, "Всё обучено на ноутбуке с видеокартой на 6 ГБ - LoRA обучает маленькие добавочные матрицы, а не всю модель.", 15, MUTED))
    slides.append({"elements": elements, "notes": "LoRA: вместо всех весов модели обучаются маленькие матрицы, потом их сливают с моделью, "
                                                  "поэтому она работает так же быстро. У большой модели обучали только декодер - иначе не хватило бы памяти."})

    # 5 - data and honest testing
    elements = [title("Данные и честная проверка")]
    sources = [("FLEURS", "прочитанные предложения, 3 языка"), ("KSC2", "казахский: чтение, эфир, парламент, подкасты"),
               ("Golos", "короткие русские фразы"), ("LibriSpeech", "английские аудиокниги - режим двух голосов")]
    for i, (name, about) in enumerate(sources):
        y = 110 + i * 62
        elements += [box(M, y, 420, 52, CARD2), text(M + 16, y, 120, 52, name, 16, GREEN, bold=True, valign="middle"),
                     text(M + 140, y, 270, 52, about, 14, TEXT, valign="middle")]
    elements.append(text(M, 365, 420, 80, "Записи со сменой языка собраны из 2-4 фраз на разных языках: открытого корпуса смешанной казахско-русско-английской речи нет.", 13, MUTED))
    flow = [("Обучение", GREEN), ("Проверка: выбор настроек", GREEN), ("Тест: открыт один раз", ORANGE)]
    for i, (label, color) in enumerate(flow):
        y = 110 + i * 80
        elements.append(box(520, y, 392, 56, color, label=label, size=17))
        if i < 2:
            elements.append({"type": "arrow", "x1": 716, "y1": y + 58, "x2": 716, "y2": y + 78, "color": MUTED})
    elements.append(text(520, 360, 392, 90, "Роли - из официального деления корпусов. Три запечатанных тестовых набора: A - малая модель, "
                                             "B - большая, C - приложение в шуме. Все решения приняты на проверке, до теста.", 14, TEXT))
    slides.append({"elements": elements, "notes": "Главное правило: тест открывается один раз, после того как всё выбрано на проверочных "
                                                  "данных. Для большой модели мы заморозили новый тестовый набор до начала обучения."})

    # 6 - English result
    slides.append({"elements": [
        title("Результат 1: два голоса одновременно (английский)"),
        text(M, 120, 380, 70, f"{pct(t2['pretrained_wer'])} → {pct(t2['cleartalk_wer'])}", 40, ORANGE, bold=True, font="Cambria"),
        text(M, 190, 380, 60, f"ошибок в словах на новых смесях двух голосов ({t2['units']} записей): на {abs(100 * t2['relative_change']):.0f}% меньше", 15, TEXT),
        text(M, 270, 380, 50, f"{pct(t3['pretrained_wer'])} → {pct(t3['cleartalk_wer'])}", 30, GREEN, bold=True, font="Cambria"),
        text(M, 320, 380, 50, "чистая речь одного человека не испортилась", 15, TEXT),
        text(M, 390, 380, 80, "Выигрыш - в основном меньше вставок: модель перестала дописывать слова второго голоса, просочившиеся через разделитель.", 13, MUTED),
        {"type": "bars", "x": 450, "y": 120, "w": W - M - 450, "h": 300, "max": 0.36, "label_width": 230, "size": 14, "rows": english_bars},
        text(450, 435, W - M - 450, 40, "Итоговый тест: дикторы, которых модель не видела; серый - готовая модель, оранжевый - наша.", 13, MUTED),
    ], "notes": "Итоговый тест английской модели: дикторы, которых модель не видела, тест открыт один раз."})

    # 7 - mixed speech
    bars = [("Whisper small (готовая)", score[("small", "mixed")], GREY, False), ("Whisper large-v3-turbo", score[("turbo", "mixed")], GREY, False),
            ("Казахская turbo (исходная)", score[("kazakh", "mixed")], GREY, False), ("Наша малая", score[("cleartalk", "mixed")], "E9B384", False),
            ("Наша большая", score[("cleartalk-large", "mixed")], ORANGE, True), ("Приложение, «языки меняются в реплике»", score[("pipeline:mixed", "mixed")], GREEN, True)]
    slides.append({"elements": [
        title("Результат 2: смена языка - ошибок в 3 раза меньше"),
        {"type": "bars", "x": M, "y": 110, "w": W - 2 * M, "h": 300, "max": 0.65, "label_width": 330, "size": 15,
         "rows": [{"label": label, "value": value, "text": pct(value), "color": color, "bold": bold} for label, value, color, bold in bars]},
        text(M, 430, W - 2 * M, 60, f"Тест B: {test_b['units'] // 6} записей со сменой языка, которых ни одна модель не видела. Доля ошибок в словах, "
                                    "меньше - лучше. Исходная модель после смены языка теряла целые фразы: пропущенных слов на проверке 2 416 → 170.", 13, MUTED),
    ], "notes": "Главный результат. Готовые модели в записи со сменой языка продолжают писать на первом языке или бросают текст. "
                "Наша большая модель этого не делает; лучший результат даёт приложение с настройкой «языки меняются в реплике»."})

    # 8 - per language
    rows = [["Набор (тест B)", "Исходная казахская turbo", "Наша большая", "Whisper large-v3-turbo"]]
    for set_name, label in (("fleurs-kk", "Казахский, чтение"), ("ksc2-kk", "Казахский, эфир (KSC2)"), ("fleurs-ru", "Русский, чтение"),
                            ("golos-ru", "Русский, короткие фразы"), ("fleurs-en", "Английский, чтение")):
        values = [score[(s, set_name)] for s in ("kazakh", "cleartalk-large", "turbo")]
        best = min(values)
        rows.append([label] + [("**" if v == best else "") + pct(v) for v in values])
    slides.append({"elements": [
        title("Результат 3: по языкам"),
        {"type": "table", "x": M, "y": 105, "w": W - 2 * M, "rows": rows, "col_widths": [300, 188, 188, 188], "size": 15, "row_height": 40,
         "header_fill": GREEN, "header_color": WHITE, "band_fill": CARD2, "strong_color": ORANGE},
        text(M, 365, W - 2 * M, 50, "Приложение (по правилу, записанному до обучения): казахский и русский → наша большая, английский → large-v3-turbo.", 15, TEXT),
        text(M, 420, W - 2 * M, 60, "Честно: на казахском эфире наша модель на тесте в пределах шума от исходной, на проверке - на 0,9 п.п. хуже. "
                                    "Английский восстановлен не полностью.", 14, MUTED),
    ], "notes": "Лучшее значение в строке выделено. Минусы называем сами, до вопроса."})

    # 9 - robustness, and the protections of round 4
    final = load("robustness-final.json")
    slides.append(robustness_final_slide(final) if final else robustness_slide(robust))
    options = {name: load(f"option-{name}.json") for name in ("background", "region", "overlap", "kk")}
    if options["background"]:
        slides.append(protections_slide(options, load("test-final-D.json")))

    # 10 - the app itself
    demo = ROOT / "docs" / "slides" / "demo_result.json"
    elements = [title("Как это выглядит")]
    if demo.is_file():
        # The real output of the packaged demo ("languages change inside turns"), drawn the way the app lists turns.
        turns = json.loads(demo.read_text(encoding="utf-8"))["turns"][:5]
        chip = {"kk": (KK, "КАЗАХСКИЙ"), "ru": (RU, "РУССКИЙ"), "en": (EN, "АНГЛИЙСКИЙ")}
        elements += [box(M, 100, 570, 390, WHITE, line="D8DED9"),
                     text(M + 18, 110, 534, 24, "РАЗГОВОР · пример «Три языка. Шесть голосов.»", 11, MUTED, bold=True)]
        row_h = 70
        for i, turn in enumerate(turns):
            y = 142 + i * row_h
            minutes = lambda value: f"{int(value) // 60}:{int(value) % 60:02d}"  # noqa: E731
            color, label = chip.get(turn["language"], (GREY, turn["language_name"].upper()))
            speaker = turn["speaker"].replace("Speaker", "Говорящий")
            elements += [box(M + 14, y, 542, row_h - 8, CARD2),
                         text(M + 26, y + 6, 86, 20, f"{minutes(turn['start'])} - {minutes(turn['end'])}", 11, ORANGE, bold=True),
                         text(M + 118, y + 6, 130, 20, speaker, 12, TEXT, bold=True),
                         box(M + 250, y + 6, 92, 18, color, label=label, size=9),
                         text(M + 26, y + 28, 520, 34, turn["text"], 12, TEXT)]
    else:
        elements.append(box(M, 100, 570, 390, CARD2, label="пример появится после расшифровки", size=16, color=MUTED))
    elements.append({"type": "bullets", "x": 650, "y": 110, "w": W - 650 - M, "h": 380, "size": 15, "space_after": 10, "bullet_color": ORANGE, "items": [
        "Каждая реплика: время (нажми - и запись играет с этого места), говорящий, язык, текст",
        "Сомнительные места помечены «Проверьте»; фоновые голоса и зацикливания в текст не попадают",
        "Говорящим можно дать имена, текст - исправить",
        "Сохранение текста и субтитров",
        "Работает на своём компьютере; можно запустить в Google Colab"]})
    slides.append({"elements": elements, "notes": "Здесь удобно переключиться на живое демо: пример «Три языка. Шесть голосов.», "
                                                  "настройка «Языки меняются в реплике». Запусти его заранее - первый запуск грузит модели."})

    # 11 - limits
    overlap_on = bool(options.get("overlap") and options["overlap"]["adopted"])
    slides.append({"elements": [
        title("Ограничения"),
        {"type": "bullets", "x": M, "y": 110, "w": 560, "h": 380, "size": 15, "space_after": 10, "bullet_color": ORANGE, "items": [
            "Смешанные записи собраны из фраз на одном языке: смена языка всегда совпадает со сменой говорящего",
            "Переключение языка посреди фразы и живые разговоры ещё не измерены",
            ("Одновременная речь разделяется, только если метки говорящих видят там двух людей; режим двух голосов - только английский"
             if overlap_on else "Режим «Разговор» не разделяет людей, говорящих одновременно (разделение проверено, но добавляло лишний текст); режим двух голосов - только английский"),
            "Отложенные «фоновые» кусочки стоит просмотреть: тихий настоящий ответ в шумной комнате может оказаться среди них",
            "«Говорящий 1, 2…» - метки внутри записи, а не установленные личности",
            "На Mac всё считается на процессоре - медленнее, чем на видеокарте (или в Google Colab)"]},
        box(650, 120, 262, 230, CARD),
        text(670, 140, 222, 30, "Как это исправить", 18, TEXT, bold=True),
        text(670, 180, 222, 230, "В приложении есть «проверочные примеры»: исправленная расшифровка своей записи сохраняется на компьютере, "
                                 "и одна команда считает ошибки каждого способа на живой речи.", 14, TEXT),
    ], "notes": "Не называть долю ошибок «точностью» и не обещать качество на реальных встречах - его ещё надо измерить."})

    # 12 - summary
    big = [(f"{pct(score[('kazakh', 'mixed')], 0)} → {pct(score[('cleartalk-large', 'mixed')], 0)}", "ошибок на записях со сменой языка"),
           (f"{pct(t2['pretrained_wer'], 0)} → {pct(t2['cleartalk_wer'], 0)}", "ошибок на смесях двух голосов")]
    test_d = load("test-final-D.json")
    if test_d and test_d.get("mode") == "app":
        noisy = test_d["condition"]["babble-10"]
        big.append((f"{pct(noisy['before'], 0)} → {pct(noisy['final'], 0)}", "ошибок на тесте, когда фоном говорят другие люди"))
    elif final:
        noisy = next((row for row in final["rows"] if row["condition"] == "babble-10" and row["set"] == "mixed"), None)
        if noisy:
            big.append((f"{pct(noisy['before'], 0)} → {pct(noisy['after'], 0)}", "ошибок, когда фоном говорят другие люди"))
    big.append(("3", "модели, дообученные на ноутбуке"))
    elements = [text(M, 50, W - 2 * M, 50, "Итог", 34, WHITE, bold=True, font="Cambria")]
    for i, (number, label) in enumerate(big):
        y = 120 + i * (300 // len(big) + 4)
        elements += [text(M, y, 250, 60, number, 38, ORANGE, bold=True, font="Cambria", valign="middle"),
                     text(M + 260, y, 290, 60, label, 16, SOFT, valign="middle")]
    elements += [
        text(650, 130, W - 650 - M, 30, "Дальше", 20, WHITE, bold=True),
        {"type": "bullets", "x": 650, "y": 170, "w": W - 650 - M, "h": 230, "size": 15, "space_after": 8, "color": SOFT, "bullet_color": ORANGE, "items": [
            "Измерить на живых записях",
            "Разделять одновременную речь, не выдумывая слов",
            "Обучить и «слух» большой модели на видеокарте на 16 ГБ"]},
        text(M, 450, W - 2 * M, 50, "Спасибо! Вопросы?", 28, WHITE, bold=True, font="Cambria"),
    ]
    slides.append({"bg": DARK, "elements": elements, "notes": "Подвести итог тремя числами и перейти к вопросам."})

    deck = {"body_font": "Calibri", "text_color": TEXT, "slides": slides}
    SPEC.parent.mkdir(parents=True, exist_ok=True)
    SPEC.write_text(json.dumps(deck, ensure_ascii=False, indent=1), encoding="utf-8")
    return deck


def main():
    deck = build()
    done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "render_slides.ps1"),
                           str(SPEC), str(OUT), str(PNG)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(done.stdout.strip(), done.stderr.strip()[-3000:])
    if done.returncode != 0:
        sys.exit(done.returncode)
    print(f"{len(deck['slides'])} slides")


if __name__ == "__main__":
    main()
