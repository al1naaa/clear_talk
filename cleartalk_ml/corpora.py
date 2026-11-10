"""Unpack the downloaded speech corpora and describe every utterance in one table.

    python -m cleartalk_ml.corpora            # unpack what is downloaded, write data/ml/utterances.csv.gz

Corpora (all public and ungated, see scripts/download_multilingual.py):

* FLEURS (en, ru, kk) - read Wikipedia-style sentences; official train / dev / test parts; a sentence
  has one id shared by all its readers and by its translations in the other languages.
* KSC2 (kk) - the beginning of the archive only: the official Test folders and the start of Train.
* Golos crowd, 10-hour subset (ru) - short crowd-sourced phrases; official train / validation / test.

Every step is restartable: finished archives leave a marker and are skipped.
"""

import argparse
import io
import tarfile
from pathlib import Path

import pandas as pd

from .paths import AUDIO, DATA, RAW, SAMPLE_RATE, rel

FLEURS_COLUMNS = ["sentence_id", "file", "text_raw", "text_norm", "chars", "num_samples", "gender"]
MANIFEST = DATA / "utterances.csv.gz"


def _marker(folder: Path) -> Path:
    return folder / ".complete"


def unpack_fleurs() -> None:
    for archive in sorted((RAW / "fleurs").glob("*/*.tar.gz")):
        language, split = archive.parent.name, archive.name.removesuffix(".tar.gz")
        target = AUDIO / "fleurs" / language
        if _marker(target / split).is_file():
            continue
        print(f"unpacking FLEURS {language}/{split} ...", flush=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(target, filter="data")
        _marker(target / split).write_text("ok\n", encoding="utf-8")


def fleurs_table() -> pd.DataFrame:
    frames = []
    for table in sorted((RAW / "fleurs").glob("*/*.tsv")):
        language, split = table.parent.name, table.stem
        folder = AUDIO / "fleurs" / language / split
        if not _marker(folder).is_file():
            continue
        frame = pd.read_csv(table, sep="\t", header=None, names=FLEURS_COLUMNS, quoting=3, dtype=str,
                            keep_default_na=False, encoding="utf-8")
        frame["path"] = [rel(folder / name) for name in frame.file]
        frame["exists"] = [(folder / name).is_file() for name in frame.file]
        frame["duration"] = frame.num_samples.astype(int) / SAMPLE_RATE
        frame["utt_id"] = [f"fleurs-{language}-{split}-{Path(name).stem}" for name in frame.file]
        frame = frame.assign(corpus="fleurs", language=language, split={"dev": "val"}.get(split, split),
                             domain="read sentences", gender=frame.gender.str.lower(),
                             group=["fleurs-" + s for s in frame.sentence_id])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def unpack_ksc2() -> None:
    """Unpack the cut-off archive; the last, incomplete file is discarded."""
    archive = RAW / "ksc2" / "ISSAI_KSC2.prefix.tar.gz"
    target = AUDIO / "ksc2"
    if not archive.is_file() or _marker(target).is_file():
        return
    print("unpacking the KSC2 prefix ...", flush=True)
    count = 0
    try:
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if not member.isfile() or ".." in Path(member.name).parts:
                    continue
                out = target / member.name
                out.parent.mkdir(parents=True, exist_ok=True)
                data = tar.extractfile(member).read()
                if len(data) != member.size:
                    break
                out.write_bytes(data)
                count += 1
                if count % 20000 == 0:
                    print(f"  {count:,} files", flush=True)
    except (tarfile.ReadError, EOFError, OSError) as error:
        print(f"  archive ends here ({type(error).__name__}); {count:,} complete files kept", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    _marker(target).write_text(f"{count}\n", encoding="utf-8")


def ksc2_table() -> pd.DataFrame:
    import soundfile as sf

    root = AUDIO / "ksc2"
    if not _marker(root).is_file():
        return pd.DataFrame()
    rows = []
    for audio in sorted(root.rglob("*.flac")):
        # One source (tv_news) was published with doubled extensions: 412833.flac.flac + 412833.txt.txt.
        doubled = audio.name.endswith(".flac.flac")
        stem = audio.name.removesuffix(".flac.flac") if doubled else audio.stem
        text = audio.with_name(stem + (".txt.txt" if doubled else ".txt"))
        if not text.is_file():
            continue
        try:
            info = sf.info(str(audio))
        except Exception:  # a truncated file at the cut
            continue
        split, domain = audio.parts[-3], audio.parts[-2]
        rows.append({"utt_id": f"ksc2-{split.lower()}-{stem}", "corpus": "ksc2", "language": "kk",
                     "split": {"dev": "val"}.get(split.lower(), split.lower()), "domain": domain,
                     "path": rel(audio), "exists": True, "duration": info.frames / info.samplerate,
                     "sample_rate": info.samplerate, "gender": "",
                     "text_raw": text.read_text(encoding="utf-8").strip(), "group": f"ksc2-{stem}"})
    return pd.DataFrame(rows)


def unpack_golos() -> None:
    import pyarrow.parquet as pq

    for shard in sorted((RAW / "golos").glob("*.parquet")):
        split = shard.name.split("-")[0]
        target = AUDIO / "golos" / split
        done = target / f".{shard.stem}.complete"
        if done.is_file():
            continue
        print(f"unpacking Golos {shard.name} ...", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        reader = pq.ParquetFile(shard)
        part = shard.name.split("-")[1]  # the files carry no names: number them by shard and row
        for batch in reader.iter_batches(batch_size=256):
            for item in batch.to_pylist():
                audio = item["audio"]
                name = f"{part}-{len(rows):05d}.wav"
                (target / name).write_bytes(audio["bytes"])
                rows.append({"file": name, "text_raw": item.get("transcription") or ""})
        pd.DataFrame(rows).to_csv(target / f"{shard.stem}.csv", index=False, encoding="utf-8")
        done.write_text(f"{len(rows)}\n", encoding="utf-8")


def golos_table() -> pd.DataFrame:
    import soundfile as sf

    rows = []
    for listing in sorted((AUDIO / "golos").glob("*/*.csv")):
        split = listing.parent.name
        for item in pd.read_csv(listing, dtype=str, keep_default_na=False, encoding="utf-8").itertuples():
            audio = listing.parent / item.file
            info = sf.info(str(audio))
            rows.append({"utt_id": f"golos-{split}-{audio.stem}", "corpus": "golos", "language": "ru",
                         "split": {"validation": "val"}.get(split, split), "domain": "crowd phrases",
                         "path": rel(audio), "exists": True, "duration": info.frames / info.samplerate,
                         "sample_rate": info.samplerate, "gender": "", "text_raw": item.text_raw,
                         "group": f"golos-{audio.stem}"})
    return pd.DataFrame(rows)


def build_manifest() -> pd.DataFrame:
    """One row per utterance of every unpacked corpus."""
    frames = [frame for frame in (fleurs_table(), ksc2_table(), golos_table()) if len(frame)]
    table = pd.concat(frames, ignore_index=True)
    columns = ["utt_id", "corpus", "language", "split", "domain", "group", "path", "exists", "duration",
               "gender", "text_raw", "text_norm", "sentence_id"]
    for column in columns:
        if column not in table:
            table[column] = ""
    return table[columns].fillna("")


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST, dtype={"sentence_id": str, "gender": str, "text_raw": str, "text_norm": str},
                       keep_default_na=False, encoding="utf-8")


def read_audio(path, sample_rate: int = SAMPLE_RATE):
    """Mono float32 at 16 kHz, decoded the way the app decodes uploads (PyAV through faster-whisper)."""
    from faster_whisper.audio import decode_audio

    from .paths import ROOT

    return decode_audio(str(ROOT / path), sampling_rate=sample_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    unpack_fleurs()
    unpack_ksc2()
    unpack_golos()
    table = build_manifest()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(MANIFEST, index=False, encoding="utf-8")
    summary = table.groupby(["corpus", "language", "split"]).agg(
        utterances=("utt_id", "size"), hours=("duration", lambda s: round(s.sum() / 3600, 2)))
    print(summary.to_string())
    print(f"{MANIFEST.relative_to(DATA.parents[1])}: {len(table):,} utterances")


if __name__ == "__main__":
    main()
