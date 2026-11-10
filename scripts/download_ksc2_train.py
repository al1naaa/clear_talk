"""More Kazakh training speech: the broadcast sources of the KSC2 training part (approved by the user on 2026-10-14).

    python scripts/download_ksc2_train.py              # stream, keep what is wanted, stop at the transfer cap

KSC2 is one gzip-compressed tar archive of 81 GB, published in ten parts. Its members come in this order: the
official Test folders, then Train/crowdsourced (the source the first download already holds a part of), then the
other Train sources. A gzip stream can only be read from its start, so the archive is read in order - the first
5 GB from the copy already on disk, the rest over the network - and only files of the wanted sources are written
to disk. Nothing else is stored: the compressed archive itself never lands on disk.

* Kept: Train/<source>/ for parliament, podcasts, radio, talkshow and tv_news, at most ``PER_SOURCE`` utterances
  (audio + transcript) each, into data/ml/audio/ksc2/ISSAI_KSC2/Train/<source>/.
* Stops when every wanted source has its quota, the archive ends, or ``CAP_BYTES`` of the archive have been read
  (about 30 GB of new transfer - what the user approved). What was and was not reached is written to
  data/ml/raw/ksc2/train-sources.json and data/ml/downloads.json.
* A dropped connection is resumed in place (HTTP range). If the process itself stops, run the command again: the
  stream starts over (gzip cannot resume mid-stream) and files already on disk are not written again.
"""
import json
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "ml" / "raw" / "ksc2"
PREFIX = RAW / "ISSAI_KSC2.prefix.tar.gz"          # the first bytes of part "aa", downloaded on 2026-10-13
TARGET = ROOT / "data" / "ml" / "audio" / "ksc2"
STATE = RAW / "train-sources.json"
REPO = "https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2/resolve/main/"
PARTS = [f"ISSAI_KSC2.tar.gz.parta{letter}" for letter in "abcdefghij"]
WANTED = ("parliament", "podcasts", "radio", "talkshow", "tv_news")
PER_SOURCE = 3000
CAP_BYTES = 35 * 10 ** 9                            # 5 GB already on disk + about 30 GB of new transfer


class ArchiveBytes:
    """The compressed archive as one stream: the local prefix, then the parts over HTTP, resumed in place."""

    def __init__(self):
        self.position = 0             # bytes of the whole archive handed out so far
        self.local = PREFIX.open("rb")
        self.part, self.offset, self.response = 0, PREFIX.stat().st_size, None
        self.sizes = {}

    def _open(self):
        name = PARTS[self.part]
        for attempt in range(40):
            try:
                request = urllib.request.Request(REPO + name, headers={"User-Agent": "cleartalk-setup", "Range": f"bytes={self.offset}-"})
                self.response = urllib.request.urlopen(request, timeout=60)
                if self.response.status == 200 and self.offset:
                    raise OSError("the server ignored the range request")
                total = self.response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                if total.isdigit():
                    self.sizes[name] = int(total)
                return
            except urllib.error.HTTPError as error:
                if error.code == 416:     # this part is finished
                    self.response = None
                    return
                problem = f"HTTP {error.code}"
            except (OSError, urllib.error.URLError) as error:
                problem = str(error)
            wait = min(120, 5 * (attempt + 1))
            print(f"    connection problem ({problem}); retry {attempt + 1} in {wait}s", flush=True)
            time.sleep(wait)
        raise SystemExit("The download keeps failing; run the same command again later.")

    def read(self, size=1 << 20):
        if self.local is not None:
            chunk = self.local.read(size)
            if chunk:
                self.position += len(chunk)
                return chunk
            self.local.close()
            self.local = None
        while self.part < len(PARTS):
            if self.response is None:
                self._open()
                if self.response is None:   # range past the end of this part
                    self.part, self.offset = self.part + 1, 0
                    continue
            try:
                chunk = self.response.read(size)
            except (OSError, urllib.error.URLError) as error:
                print(f"    connection dropped ({error}); resuming at byte {self.offset:,}", flush=True)
                self.response = None
                continue
            if chunk:
                self.offset += len(chunk)
                self.position += len(chunk)
                return chunk
            self.response = None
            name = PARTS[self.part]
            if name in self.sizes and self.offset < self.sizes[name]:
                continue                     # closed early: resume this part
            self.part, self.offset = self.part + 1, 0
        return b""


class Decompressed:
    """File-like view of the gunzipped archive, for tarfile's streaming mode."""

    def __init__(self, raw):
        self.raw, self.buffer = raw, b""
        self.inflate = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def read(self, size=-1):
        while size < 0 or len(self.buffer) < size:
            chunk = self.raw.read()
            if not chunk:
                self.buffer += self.inflate.flush()
                break
            self.buffer += self.inflate.decompress(chunk)
            if self.raw.position >= CAP_BYTES:
                raise CapReached
        if size < 0:
            out, self.buffer = self.buffer, b""
        else:
            out, self.buffer = self.buffer[:size], self.buffer[size:]
        return out


class CapReached(Exception):
    pass


def main():
    if not PREFIX.is_file():
        raise SystemExit(f"{PREFIX} is missing: run scripts/download_multilingual.py --group ksc2 first")
    started = time.monotonic()
    raw = ArchiveBytes()
    counts = {source: 0 for source in WANTED}
    kept = set()                      # (source, utterance id) whose audio was kept: their transcripts are kept too
    first_seen, last_member, seen_sources = {}, None, []
    status = "archive ended"
    try:
        with tarfile.open(fileobj=Decompressed(raw), mode="r|") as tar:
            for member in tar:
                parts = Path(member.name).parts
                last_member = member.name
                if len(parts) >= 3 and parts[1] == "Train" and parts[2] not in seen_sources:
                    seen_sources.append(parts[2])
                    first_seen[parts[2]] = raw.position
                    print(f"  Train/{parts[2]} begins at about {raw.position / 1e9:.1f} GB of the archive", flush=True)
                if not member.isfile() or ".." in parts or len(parts) < 4 or parts[1] != "Train" or parts[2] not in WANTED:
                    continue
                source = parts[2]
                is_audio = member.name.endswith(".flac")
                key = (source, parts[-1].split(".")[0])   # tv_news has doubled extensions: 412833.flac.flac
                if counts[source] >= PER_SOURCE and key not in kept:
                    if is_audio and all(counts[s] >= PER_SOURCE for s in WANTED):
                        status = "every source has its quota"
                        break
                    continue
                out = TARGET / member.name
                data = tar.extractfile(member).read()
                if len(data) != member.size:
                    break
                if not out.is_file() or out.stat().st_size != member.size:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    temporary = out.with_name(out.name + ".tmp")
                    temporary.write_bytes(data)
                    temporary.replace(out)
                if is_audio:
                    kept.add(key)
                    counts[source] += 1
                    total = sum(counts.values())
                    if total % 500 == 0:
                        rate = (raw.position - PREFIX.stat().st_size) / max(time.monotonic() - started, 1) / 1e6
                        print(f"  {total:,} utterances kept {counts} · {raw.position / 1e9:.1f} GB read · {rate:.1f} MB/s", flush=True)
    except CapReached:
        status = f"transfer cap reached ({CAP_BYTES / 1e9:.0f} GB of the archive)"
    except (tarfile.ReadError, EOFError, zlib.error) as error:
        status = f"archive stream ended ({type(error).__name__})"
    summary = {"status": status, "archive_bytes_read": raw.position, "new_bytes_transferred": raw.position - PREFIX.stat().st_size,
               "kept_utterances": counts, "train_sources_seen": seen_sources,
               "source_starts_at_archive_byte": first_seen, "last_member": last_member,
               "not_reached": [s for s in WANTED if s not in seen_sources], "per_source_quota": PER_SOURCE,
               "minutes": round((time.monotonic() - started) / 60, 1), "date": time.strftime("%Y-%m-%d %H:%M")}
    STATE.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    log_path = ROOT / "data" / "ml" / "downloads.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else {}
    log["data/ml/audio/ksc2/ISSAI_KSC2/Train (broadcast sources)"] = {
        "repo": "issai/Kazakh_Speech_Corpus_2", "path": "ISSAI_KSC2.tar.gz.parta* (streamed, only Train broadcast sources kept)",
        "bytes": raw.position - PREFIX.stat().st_size, "kept_utterances": counts, "status": status,
        "approved": "the user, 2026-10-14 (about 30 GB of transfer)", "downloaded": summary["date"]}
    log_path.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    print("KSC2_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
