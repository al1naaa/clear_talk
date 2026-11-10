"""Download the public models and speech data of the multilingual ClearTalk work - resumable.

    python scripts/download_multilingual.py --group eval      # FLEURS dev + test (en, ru, kk)
    python scripts/download_multilingual.py --group models    # ready recognizers + speaker embeddings
    python scripts/download_multilingual.py --group train     # FLEURS train (en, ru, kk)
    python scripts/download_multilingual.py --group golos     # Russian: Golos crowd, 10-hour subset
    python scripts/download_multilingual.py --group ksc2      # Kazakh: the first 5 GB of KSC2
    python scripts/download_multilingual.py --list            # show what each group contains
    python scripts/download_multilingual.py --relog           # record files that are on disk but not in the log

Only the standard library is used, so it runs before any environment exists. Every file is
written to ``<name>.part`` and renamed when complete; an interrupted run continues from the
bytes already on disk (HTTP Range). Nothing here needs an account or a token: all sources are
ungated. What was downloaded, from which repository commit and with which SHA-256 is recorded
in ``data/ml/downloads.json``.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.getenv("CLEARTALK_MODELS_DIR") or ROOT / "models").expanduser() / "multilingual"
RAW = ROOT / "data" / "ml" / "raw"
LOG = ROOT / "data" / "ml" / "downloads.json"
HUB = "https://huggingface.co"
LANGUAGES = {"en": "en_us", "ru": "ru_ru", "kk": "kk_kz"}
WHISPER_FILES = ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json",
                 "special_tokens_map.json", "model.safetensors"]


def fleurs(splits):
    for code, name in LANGUAGES.items():
        for split in splits:
            for path in (f"data/{name}/{split}.tsv", f"data/{name}/audio/{split}.tar.gz"):
                yield ("dataset", "google/fleurs", path, RAW / "fleurs" / code / Path(path).name)


# KSC2 is one 81 GB archive split into parts; only its beginning is taken (the test folders
# come first, then training speech). The cut-off archive is unpacked by cleartalk_ml.corpora.
KSC2_PREFIX_BYTES = 5 * 1024 ** 3
GOLOS_FILES = ["data/train-00000-of-00003-85be0fb4b3de87ab.parquet", "data/train-00001-of-00003-fcf1367484d8316d.parquet",
               "data/train-00002-of-00003-8e44cef5e8ee83d1.parquet", "data/validation-00000-of-00001-58253c41d3f63473.parquet",
               "data/test-00000-of-00003-cad0cfaddbc8fa71.parquet", "data/test-00001-of-00003-9d9aa103f1ceb4c7.parquet",
               "data/test-00002-of-00003-91f7e0c1d8f0af36.parquet"]


def whisper(repo, folder, optional=()):
    for name in WHISPER_FILES:
        yield ("model", repo, name, MODELS / folder / name, name in optional)


GROUPS = {
    "eval": lambda: list(fleurs(["dev", "test"])),
    "train": lambda: list(fleurs(["train"])),
    "models": lambda: [
        *whisper("openai/whisper-large-v3-turbo", "whisper-large-v3-turbo-hf"),
        *whisper("shyngys879/kazakh-whisper-large-v3-turbo", "kazakh-whisper-large-v3-turbo-hf",
                 optional=("vocab.json", "merges.txt", "normalizer.json", "added_tokens.json", "special_tokens_map.json")),
        *whisper("openai/whisper-small", "whisper-small-hf"),
        *[("model", "speechbrain/spkrec-ecapa-voxceleb", name, MODELS / "spkrec-ecapa-voxceleb" / name)
          for name in ("hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt",
                       "label_encoder.txt")],
    ],
    "golos": lambda: [("dataset", "bond005/sberdevices_golos_10h_crowd", path, RAW / "golos" / Path(path).name)
                      for path in GOLOS_FILES],
    "ksc2": lambda: [("dataset", "issai/Kazakh_Speech_Corpus_2", "ISSAI_KSC2.tar.gz.partaa",
                      RAW / "ksc2" / "ISSAI_KSC2.prefix.tar.gz", False, KSC2_PREFIX_BYTES)],
}


def url_of(kind, repo, path):
    prefix = "datasets/" if kind == "dataset" else ""
    return f"{HUB}/{prefix}{repo}/resolve/main/{path}"


def fetch(url, target, limit=None, attempts=30):
    """Download ``url`` (or its first ``limit`` bytes) to ``target`` with resume.

    Returns (bytes, commit), or None when the file does not exist.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    commit = None
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if limit and have >= limit:
            part.replace(target)
            return have, commit
        request = urllib.request.Request(url, headers={"User-Agent": "cleartalk-setup",
                                                       "Range": f"bytes={have}-{limit - 1 if limit else ''}"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                commit = response.headers.get("X-Repo-Commit") or commit
                if response.status == 200 and have:  # the server ignored Range: start again
                    have = 0
                    part.unlink()
                total = have + int(response.headers.get("Content-Length") or 0)
                shown = time.monotonic()
                with part.open("ab") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        have += len(chunk)
                        if time.monotonic() - shown > 30:
                            print(f"    {have / 1e6:8.0f} / {total / 1e6:.0f} MB", flush=True)
                            shown = time.monotonic()
            if total and have < total:
                raise OSError(f"connection closed at {have} of {total} bytes")
            part.replace(target)
            return have, commit
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code == 416 and have:  # the part file is already complete
                part.replace(target)
                return have, commit
            problem = f"HTTP {error.code}"
        except (OSError, urllib.error.URLError) as error:
            problem = str(error)
        wait = min(120, 5 * (attempt + 1))
        print(f"    interrupted ({problem}); retry {attempt + 1}/{attempts} in {wait}s", flush=True)
        time.sleep(wait)
    raise SystemExit(f"Could not download {url}; run the same command again to continue.")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_log():
    return json.loads(LOG.read_text(encoding="utf-8")) if LOG.is_file() else {}


def record(key, entry):
    """Add one entry to the log, re-reading the file first so that two downloads running at the same
    time do not overwrite each other's entries."""
    log = read_log()
    log[key] = entry
    LOG.parent.mkdir(parents=True, exist_ok=True)
    temporary = LOG.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(sorted(log.items())), indent=1) + "\n", encoding="utf-8")
    temporary.replace(LOG)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=sorted(GROUPS), action="append")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--relog", action="store_true")
    args = parser.parse_args()
    groups = args.group or sorted(GROUPS)
    log = read_log()
    for group in groups:
        for kind, repo, path, target, *rest in GROUPS[group]():
            optional = bool(rest and rest[0])
            limit = rest[1] if len(rest) > 1 else None
            key = target.relative_to(ROOT).as_posix()
            if args.list:
                print(f"{group:7s} {repo}/{path} -> {key}")
                continue
            if args.relog:
                if target.is_file() and key not in log:
                    record(key, {"repo": repo, "path": path, "commit": None, "bytes": target.stat().st_size,
                                 "sha256": sha256(target), **({"first_bytes_only": limit} if limit else {}),
                                 "note": "entry rebuilt from the file on disk; the repository commit was not recorded"})
                    print(f"recorded {key}", flush=True)
                continue
            if target.is_file() and key in read_log():
                continue
            print(f"[{group}] {repo}/{path}", flush=True)
            result = fetch(url_of(kind, repo, path), target, limit)
            if result is None:
                if optional:
                    continue
                raise SystemExit(f"{repo}/{path} was not found")
            size, commit = result
            record(key, {"repo": repo, "path": path, "commit": commit, "bytes": size, "sha256": sha256(target),
                         **({"first_bytes_only": limit} if limit else {}),
                         "downloaded": time.strftime("%Y-%m-%d %H:%M")})
    if not args.list and not args.relog:
        print(f"done: {', '.join(groups)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
