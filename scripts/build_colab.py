"""Generate notebooks/ClearTalk_ASR_Colab.ipynb - the same pipeline on a Colab GPU.

    python scripts/build_colab.py

The notebook calls the project's own modules (no copied logic), so a Colab run and a local
run differ only in the device. It was generated and checked for syntax here; it has not been
executed in Colab by the authors of this repository.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("markdown", """# ClearTalk · fine-tune Whisper small.en on separated speech (Colab GPU)

Reproduces the local experiment on a Colab GPU: data → two-speaker mixtures → frozen SepFormer →
LoRA fine-tuning of Whisper small.en → validation → export for the app.

1. **Runtime → Change runtime type → GPU** (T4 or better).
2. Run the cells from top to bottom. Expect several hours in total and ~25 GB of disk.
3. Upload `dist/ClearTalk_source.zip` (made locally with `python scripts/package_source.py`).

The test sets are **not** scored here: like the local run, the one-time final test belongs to
`ml_pipeline_english/04_eval.py` after the validation decision."""),
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", """from google.colab import files
import zipfile, os
uploaded = files.upload()  # choose ClearTalk_source.zip
zipfile.ZipFile(next(iter(uploaded))).extractall('/content/cleartalk')
os.chdir('/content/cleartalk')
print(sorted(os.listdir('.')))"""),
    ("code", """# Keep Colab's CUDA build of torch/torchaudio; pin everything else.
!grep -v -E '^(torch|torchaudio)==' requirements-train.txt > /tmp/requirements-colab.txt
!pip install -q -r /tmp/requirements-colab.txt"""),
    ("code", """import json, pathlib, subprocess, tarfile, urllib.request
root = pathlib.Path('/content/cleartalk')
downloads = root / 'data/asr/downloads'
downloads.mkdir(parents=True, exist_ok=True)
openslr = 'https://www.openslr.org/resources'

def fetch(url, path):
    if not path.exists():
        subprocess.run(['curl', '-sSL', '--fail', '-o', str(path), url], check=True)

targets = {'train-clean-100': 'data/asr', 'dev-clean': 'data/asr', 'test-clean': 'data/asr',
           'test-other': 'data/asr', 'dev-other': 'data/v4'}
for name, destination in targets.items():
    archive = downloads / f'{name}.tar.gz'
    fetch(f'{openslr}/12/{name}.tar.gz', archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root / destination)
    print('extracted', name)
pc = downloads / 'librispeech_pc_manifests.tar.gz'
fetch(f'{openslr}/145/manifests.tar.gz', pc)
(root / 'data/asr/librispeech_pc').mkdir(parents=True, exist_ok=True)
with tarfile.open(pc) as tar:
    tar.extractall(root / 'data/asr/librispeech_pc')
manifest = json.loads((root / 'data/v4/esc10/manifest.json').read_text())
for clip in manifest['clips']:
    path = root / clip['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(clip['url'], path)
print('ESC-10 clips:', len(manifest['clips']))"""),
    ("code", "!python scripts/download_models.py --training"),
    ("markdown", "Steps 1-3: utterance table, profile and EDA figures (`ml_pipeline_english/figures`)."),
    ("code", "!python ml_pipeline_english/01_eda.py"),
    ("markdown", "Step 5: render the frozen recipes in `data/asr/plans/` and separate them with SepFormer on the GPU."),
    ("code", """!python -m cleartalk_asr.mixtures build --set val --device cuda --worker colab
!python -m cleartalk_asr.mixtures build --set train --device cuda --worker colab"""),
    ("markdown", "Steps 4-8 summary (uses the validation baseline in `data/asr/results/`, included in the zip)."),
    ("code", "!python ml_pipeline_english/02_prep.py"),
    ("markdown", "Steps 9-12: LoRA fine-tuning with validation-based early stopping. On CUDA, bfloat16 autocast is safe and faster."),
    ("code", "!python -m cleartalk_asr.train --run colab --device cuda --precision bf16"),
    ("code", """!python -m cleartalk_asr.export --adapter models/asr-runs/colab/best --output models/whisper-small.en-cleartalk
!cd models && zip -qr /content/whisper-small.en-cleartalk.zip whisper-small.en-cleartalk
files.download('/content/whisper-small.en-cleartalk.zip')"""),
    ("markdown", """Unzip the download into the local project's `models/` folder, then run
`scripts/activate_model.py` there to compare it with the pretrained recognizer on validation."""),
]


def main():
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                             "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks" / "ClearTalk_ASR_Colab.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
