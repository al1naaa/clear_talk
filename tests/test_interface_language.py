"""The Russian interface: every sentence a user can see has a translation, and speech is left alone.

web/i18n.js translates the page as it is drawn. A sentence added to the page, to app.js or to the server
without a translation would silently stay English; these tests collect the sentences from the sources
and fail on the first one that has no Russian.
"""

import ast
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is needed to run the interface script")

# Text that is the same in both languages: names, badges and marks.
SAME = {"ClearTalk", "MIX", "A", "B", "LibriSpeech ↗", "RU"}
# Beginnings of sentences that app.js completes with numbers; the whole sentences are checked below.
BEGINNINGS = {"Validation, two-speaker mixtures", "Validation, single clean speaker", "Final test, two-speaker mixtures",
              "Final test, single clean speaker"}


def translations(texts):
    """Russian for each text (None where web/i18n.js has no translation)."""
    script = "const t=require(process.argv[1]).translate;const a=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(a.map(x=>t(x))));"
    done = subprocess.run([NODE, "-e", script, str(WEB / "i18n.js")], input=json.dumps(texts), capture_output=True,
                          text=True, encoding="utf-8", check=True)
    return dict(zip(texts, json.loads(done.stdout)))


def untranslated(texts):
    texts = sorted({" ".join(text.split()) for text in texts if re.search(r"[A-Za-z]", text)} - SAME)
    return [text for text, russian in translations(texts).items() if russian is None]


class PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.texts, self.skipping = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skipping += 1
        for name, value in attrs:
            if name in ("placeholder", "aria-label", "title") and value:
                self.texts.append(value)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skipping -= 1

    def handle_data(self, data):
        if not self.skipping and data.strip():
            self.texts.append(data)


def test_every_sentence_of_the_page_has_russian():
    page = PageText()
    page.feed((WEB / "index.html").read_text(encoding="utf-8"))
    assert len(page.texts) > 120
    assert untranslated(page.texts) == []


def test_every_fixed_sentence_of_the_script_has_russian():
    """Sentences app.js puts on the page: quoted strings that read like text, not like code."""
    source = (WEB / "app.js").read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"^\s*//.*$", "", source, flags=re.M)
    strings = re.findall(r"'((?:[^'\\\n]|\\.)*)'", source)
    sentences = [text for text in strings if re.match(r"[A-Z(\[]", text) and " " in text.strip() and not text.startswith("Content-")]
    assert len(sentences) > 60
    assert untranslated(set(sentences) - BEGINNINGS) == []
    whole = [f"{beginning}: word error rate 21.4% pretrained → 12.9% ClearTalk on the same 1,000 mixtures. Lower is better."
             for beginning in sorted(BEGINNINGS)]
    assert untranslated(whole) == []


def server_sentences():
    """Fixed sentences the server sends to the page: errors, progress stages and notes on a result."""
    shown_by = {"HTTPException", "AudioValidationError", "update", "_no_speech_result", "append", "extend"}
    found = []
    for path in sorted((ROOT / "backend").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        pieces = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for part in node.values}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                target = getattr(getattr(node.func, "value", None), "id", "")
                if name not in shown_by or (name in ("append", "extend") and target not in ("warnings", "review_notes")):
                    continue
                stage = name == "update" and target == ""
                for argument in ast.walk(node):
                    if id(argument) in pieces:  # a sentence built with numbers or names: covered by the samples below
                        continue
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and " " in argument.value:
                        text = argument.value
                        if stage:  # the page shows stages without hyphens, see showProgress in app.js
                            text = re.sub(r"[_-]+", " ", text)
                        found.append(text)
            elif isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "warnings" for target in node.targets):
                found += [item.value for item in ast.walk(node.value) if isinstance(item, ast.Constant) and isinstance(item.value, str)]
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple) and any(
                    getattr(element, "id", "") == "message" for target in node.targets for element in getattr(target, "elts", [])):
                found.append(node.value.elts[0].value)  # message, code = "...", "..."
            elif isinstance(node, ast.keyword) and node.arg in ("stage", "error") and isinstance(node.value, ast.Constant):
                found.append(node.value.value)
    return [text for text in found if re.match(r"[A-Z]", text)]


def test_every_fixed_sentence_of_the_server_has_russian():
    sentences = server_sentences()
    assert len(sentences) > 70 and "Detecting the language of each turn" in sentences
    assert any(text.startswith("Speaker 1, Speaker 2") for text in sentences)
    assert untranslated(sentences) == []


def test_sentences_with_numbers_and_names_are_translated():
    made = translations([
        "Kazakh → Kazakh Whisper large-v3-turbo; Russian → Whisper large-v3-turbo; English → Whisper large-v3-turbo. The language is detected turn by turn.",
        "Every turn → ClearTalk large multilingual (LoRA fine-tuned). Use this when people switch language in the middle of a turn.",
        "ClearTalk multilingual (LoRA fine-tuned) (ours) · not installed",
        "3 speakers · 21 turns · Kazakh 0:42, Russian 1:05, Kazakh + English 0:09",
        "Speaker 2 · 0:42 · Kazakh, Russian",
        "Transcribing turn 3 of 41",
        "Please trim the recording to 60 minutes or less.",
        "Please choose a file smaller than 300 MB.",
        "Whisper large-v3-turbo is not installed. Select another recognizer.",
        "Validation, mixed-language recordings (300): word error rate 60.0% Whisper small → 16.1% ClearTalk. Lower is better.",
        "Final test, Kazakh broadcast and read speech (300): 9.9% Kazakh Whisper large-v3-turbo → 9.1% ClearTalk large.",
        "Validation, mixed-language recordings (300): 16.1% ClearTalk small → 9.8% ClearTalk large.",
        "2,129,920 trained LoRA parameters (0.3% of 811,008,000); merged, so it runs as fast as Whisper large-v3-turbo.",
        "Check 1.2-3.4s: an ASR score crossed a review threshold. This is a heuristic, not a calibrated confidence estimate.",
        "Saved. Check examples on this computer: 4.",
        "Name for Speaker 2 (leave empty to go back to the label)",
    ])
    assert all(russian and not re.search(r"\b(the|is|of|or|and|turn|speaker)\b", russian) for russian in made.values()), made
    assert made["3 speakers · 21 turns · Kazakh 0:42, Russian 1:05, Kazakh + English 0:09"] == \
        "3 говорящих · 21 реплика · Казахский 0:42, Русский 1:05, Казахский + Английский 0:09"
    assert made["Transcribing turn 3 of 41"] == "Расшифровываем реплику 3 из 41"


def test_speech_names_and_files_are_never_translated():
    made = translations(["привет world, let's talk", "Сәлем, this is a mixed turn", "meeting 12 may.wav", "Aliya", "ClearTalk",
                         "Whisper large-v3-turbo", "so this is English speech about the Language of this turn and more"])
    assert set(made.values()) == {None}


def test_the_page_loads_the_language_layer_before_the_interface_script():
    page = (WEB / "index.html").read_text(encoding="utf-8")
    assert page.index('src="/static/i18n.js"') < page.index('src="/static/app.js"')
    assert 'id="lang-switch"' in page
    layer = (WEB / "i18n.js").read_text(encoding="utf-8")
    # recognized speech and typed names sit in these places; the layer must keep away from them
    for place in (".turn-text", ".transcript:not(.unavailable)", '[contenteditable="true"]', '[translate="no"]'):
        assert place in layer
