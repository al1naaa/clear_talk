"""Generate notebooks/ClearTalk_Colab_App.ipynb - the ClearTalk app on a Google Colab graphics card.

    python scripts/build_colab_app.py

The notebook unpacks the two archives written by scripts/package_source.py and scripts/package_models.py (and,
if present, the English-mode archive of ``scripts/package_models.py --english``) from the user's Google Drive,
installs the app's requirements next to Colab's own CUDA build of PyTorch, starts the app and opens it in a
separate browser tab through Colab's own proxy. That address works only for the signed-in Colab user: nothing is
published. A fallback cell transcribes an uploaded recording without the web page.

It was generated and checked here (JSON and Python syntax, the cells' logic on a local copy); it can only be run
for real in Colab, from the user's own Google account.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("markdown", """# ClearTalk в Google Colab

Приложение ClearTalk работает на видеокарте Google, а твой компьютер только показывает страницу.
Ничего не публикуется: страница открывается по ссылке, которая работает только под твоим входом в Google.

**Один раз заранее**
1. Собери архивы (на Acer): `scripts\\package_source.py` и `scripts\\package_models.py`. Для английского режима
   с двумя голосами - ещё `scripts/package_models.py --english` на Mac (там лежит дообученная английская модель).
2. Создай на Google Диске папку **ClearTalk** (в «Моём диске») и положи туда `ClearTalk_source.zip`,
   `ClearTalk_multilingual_models.zip` и, по желанию, `ClearTalk_english_models.zip`. Нужно около 7 ГБ на Диске.

**Каждый раз**
1. Меню **Среда выполнения → Сменить среду выполнения → T4 GPU**.
2. Запусти ячейки по порядку (до готовности около 5 минут).
3. Открой ссылку, которую покажет ячейка «Открыть приложение».

Когда сеанс Colab закрывается (или долго простаивает), записи и результаты исчезают - сохраняй текст и субтитры
кнопкой «Сохранить текст». Бесплатный Colab не всегда даёт видеокарту: перед защитой запусти блокнот заранее."""),
    ("code", """# 1. Есть ли видеокарта
import subprocess
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
print(gpu.stdout.strip() or 'Видеокарты нет: Среда выполнения → Сменить среду выполнения → T4 GPU, потом запусти эту ячейку снова.')"""),
    ("code", """# 2. Архивы с Google Диска
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
FOLDER = Path('/content/drive/MyDrive/ClearTalk')
SOURCE, MODELS, ENGLISH = (FOLDER / name for name in ('ClearTalk_source.zip', 'ClearTalk_multilingual_models.zip', 'ClearTalk_english_models.zip'))
missing = [path.name for path in (SOURCE, MODELS) if not path.is_file()]
assert not missing, f'На Диске в папке {FOLDER} нет: {", ".join(missing)}'
print('найдено:', ', '.join(path.name for path in (SOURCE, MODELS, ENGLISH) if path.is_file()))"""),
    ("code", """# 3. Распаковка (повторный запуск ничего не делает)
import zipfile
ROOT = Path('/content/cleartalk')
for archive, marker in ((SOURCE, ROOT / 'backend' / 'app.py'), (MODELS, ROOT / 'models' / 'multilingual' / 'profile.json'),
                        (ENGLISH, ROOT / 'models' / 'candidates-v3' / 'whisper-small.en' / 'model.bin')):
    if archive.is_file() and not marker.is_file():
        print('распаковываю', archive.name, '...')
        zipfile.ZipFile(archive).extractall(ROOT)
print('готово:', sorted(p.name for p in (ROOT / 'models' / 'multilingual').iterdir() if p.is_dir()))"""),
    ("code", """# 4. Библиотеки приложения (PyTorch с видеокартой у Colab свой - его не трогаем)
import sys
requirements = [line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
                if line.strip() and not line.startswith(('#', 'torch', 'torchaudio', 'pytest'))]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *requirements], check=True)
import torch
print('PyTorch видит видеокарту:', torch.cuda.is_available())"""),
    ("code", """# 5. Запуск приложения
import json, os, time, urllib.request
env = dict(os.environ, CLEARTALK_DEVICE='cuda', CLEARTALK_DATA_DIR=str(ROOT / 'data' / 'runtime'), PYTHONUTF8='1')
log = open('/content/cleartalk_server.log', 'w')
server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1', '--port', '8765'],
                          cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
health = None
for _ in range(90):
    try:
        health = json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=2))
        break
    except Exception:
        time.sleep(2)
if health is None:
    print(open('/content/cleartalk_server.log').read()[-3000:])
    raise RuntimeError('Приложение не запустилось - выше последние строки его журнала.')
print('приложение:', health['status'], '| режим «Разговор»:', health['features']['conversation_mode'],
      '| английский режим:', health['features']['two_speaker_separation'])"""),
    ("code", """# 6. Открыть приложение - ссылка работает только под твоим входом в Google
from google.colab import output
output.serve_kernel_port_as_window(8765, path='/', anchor_text='Открыть ClearTalk в новой вкладке')"""),
    ("markdown", """**Если ссылка не открылась** - можно расшифровать запись прямо здесь, без страницы приложения.
`LANGUAGE`: `'auto'` - язык по репликам, `'mixed'` - языки меняются в реплике, `'kk'`, `'ru'` или `'en'`."""),
    ("code", """LANGUAGE = 'auto'
from google.colab import files
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['CLEARTALK_DEVICE'] = 'cuda'
from faster_whisper.audio import decode_audio
from backend.config import Settings
from backend.conversation import ConversationPipeline
uploaded = files.upload()
name = next(iter(uploaded))
result = ConversationPipeline(Settings()).transcribe(decode_audio(name, sampling_rate=16000), language=LANGUAGE, model='auto')
clock = lambda s: f'{int(s) // 60}:{int(s) % 60:02d}'
lines = [f"[{clock(t['start'])}-{clock(t['end'])}] {t['speaker']} ({t['language_name']}): {t['text']}" for t in result['turns']]
print('\\n'.join(lines))
Path('/content/cleartalk-transcript.txt').write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
files.download('/content/cleartalk-transcript.txt')"""),
    ("code", """# Остановить приложение (иначе оно работает, пока открыт сеанс)
server.terminate()"""),
]


def main():
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks" / "ClearTalk_Colab_App.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
