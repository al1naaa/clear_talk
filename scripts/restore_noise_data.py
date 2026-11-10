"""Restore public FLEURS train/dev; verify original hashes. No training or test download."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_multilingual import fetch

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def main():
    saved = json.loads((ROOT / "data/ml/downloads.json").read_text())
    selected = {key: entry for key, entry in saved.items()
                if key.startswith("data/ml/raw/fleurs/")
                and Path(key).name in {"train.tsv", "train.tar.gz", "dev.tsv", "dev.tar.gz"}}
    if len(selected) != 12:
        raise RuntimeError("Expected three languages with train/dev metadata and audio")
    receipts = []
    for key, entry in sorted(selected.items()):
        if entry["repo"] != "google/fleurs" or not entry["path"].startswith("data/"):
            raise ValueError("Unexpected source in saved manifest")
        path = ROOT / key
        if not path.is_file():
            url = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/{entry['path']}"
            print(f"Restoring {key} ({entry['bytes'] / 1e6:.0f} MB)", flush=True)
            if fetch(url, path, attempts=3) is None:
                raise RuntimeError(f"Pinned source unavailable: {key}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch; refusing to use {key}")
        receipts.append({"path": key, "sha256": actual, "bytes": path.stat().st_size})
        if path.name.endswith(".tar.gz"):
            split = path.name.removesuffix(".tar.gz")
            target = ROOT / "data/ml/audio/fleurs" / path.parent.name
            marker = target / split / ".complete"
            if not marker.is_file():
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(path, "r:gz") as archive:
                    for member in archive:
                        if member.issym() or member.islnk():
                            raise RuntimeError("Unexpected link in speech archive")
                        archive.extract(member, target, filter="data")
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(actual + "\n")
        print(f"Verified {key}", flush=True)
    output = ROOT / "ml_pipeline/noise_followup_20261110/restore_receipt.json"
    output.write_text(json.dumps({"source_revision": REVISION, "files": receipts,
                                 "test_downloaded": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
