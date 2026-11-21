"""Rebuild the exact Kazakh and Russian control inventory from public distribution files."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.control_sources import KSD_URL, RangeReader
from scripts.download_multilingual import fetch

CONTROL = ROOT / "data/ml/noise_controls"
METADATA = CONTROL / "ksd_train_metadata.json"
GOLOS = CONTROL / "test-00000-of-00001-7fb58d4cda8ff9fe.parquet"
GOLOS_REV = "c93949f7140beef4adc404e7b54841e957f81c54"
CHECKSUMS = {
    METADATA: "831d176201ed0de601874065362355eed40f4ef9f5098d27ee5338f16c3653f5",
    GOLOS: "b6708fc8ae88ee4be2ac06e48af44b1d3e5ebd1d4c7128f7abcf363f97a4b8eb",
}


def verify(path):
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != CHECKSUMS[path]:
            raise ValueError(f"Unexpected checksum for {path.name}")


def ksd_subset():
    verify(METADATA)
    rows = json.loads(METADATA.read_text(encoding="utf-8-sig"))
    ranked = sorted(
        rows, key=lambda r: hashlib.sha256(("20261110:" + r["wav"]).encode()).digest()
    )
    groups = set()
    members = set()
    for row in ranked:
        member = Path(row["wav"].removeprefix("dataset/"))
        group = member.parent / member.stem.split("_")[0]
        if group in groups or not row["text"].strip():
            continue
        members.add(member.as_posix())
        groups.add(group)
        if len(members) == 28:
            break
    assert len(members) == 28
    return [row for row in rows if row["wav"].removeprefix("dataset/") in members]


def prepare():
    CONTROL.mkdir(parents=True, exist_ok=True)
    (CONTROL / "ksd").mkdir(exist_ok=True)
    with zipfile.ZipFile(RangeReader(KSD_URL)) as archive:
        if not METADATA.exists():
            METADATA.write_bytes(archive.read("audio2/train.json"))
        for row in ksd_subset():
            member = row["wav"].removeprefix("dataset/")
            target = (
                CONTROL
                / "ksd"
                / (hashlib.sha256(member.encode()).hexdigest()[:20] + ".wav")
            )
            data = target.read_bytes() if target.exists() else archive.read(member)
            info = archive.getinfo(member)
            if len(data) != info.file_size or zipfile.crc32(data) != info.CRC:
                raise ValueError(f"Archive checksum differs for {member}")
            if not target.exists():
                target.write_bytes(data)
    if not GOLOS.exists():
        fetch(
            f"https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield/resolve/{GOLOS_REV}/data/{GOLOS.name}",
            GOLOS,
            attempts=3,
        )
    verify(GOLOS)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
